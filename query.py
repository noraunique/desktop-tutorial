"""Query interface for the Notes Q&A system."""

import os
import sys
import click
import subprocess
import glob
from colorama import init, Fore, Style

from retrieval import query_notes, format_search_result
from config import NOTES_DIR, INDEX_DIR, CLI_COLORS, SUPPORTED_EXTENSIONS

# Initialize colorama for Windows
init()


def check_and_rebuild_index(course):
    """Check if index needs rebuilding and rebuild if necessary."""
    index_file = os.path.join(INDEX_DIR, f"{course}_index.faiss")
    metadata_file = os.path.join(INDEX_DIR, f"{course}_metadata.pkl")
    course_dir = os.path.join(NOTES_DIR, course)
    
    # If index doesn't exist, build it
    if not os.path.exists(index_file) or not os.path.exists(metadata_file):
        click.echo(f"No index found. Building index for course: {course}")
        rebuild_index(course)
        return
    
    # Check if any files in notes directory are newer than the index
    if os.path.exists(course_dir):
        index_time = os.path.getmtime(index_file)
        
        # Check all supported file types
        for ext in SUPPORTED_EXTENSIONS:
            pattern = os.path.join(course_dir, f"*{ext}")
            for file_path in glob.glob(pattern):
                if os.path.getmtime(file_path) > index_time:
                    click.echo(f"New files detected. Rebuilding index for course: {course}")
                    rebuild_index(course)
                    return

def rebuild_index(course):
    """Rebuild the index for the given course."""
    try:
        result = subprocess.run([
            sys.executable, "build_index.py", 
            "--course", course, "--force"
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode == 0:
            click.echo("Index rebuilt successfully!")
        else:
            click.echo(f"Error rebuilding index: {result.stderr}")
            sys.exit(1)
    except Exception as e:
        click.echo(f"Error rebuilding index: {str(e)}")
        sys.exit(1)

@click.command()
@click.option('--course', '-c', required=True, help='Course name to query')
@click.option('--interactive', '-i', is_flag=True, help='Interactive mode for multiple queries')
@click.option('--show-snippets', '-s', is_flag=True, default=True, help='Show source snippets')
@click.argument('question', required=False)
def main(course, interactive, show_snippets, question):
    """Query your notes using natural language."""
    
    # Automatically check and rebuild index if needed
    check_and_rebuild_index(course)
    
    # Verify index exists after potential rebuild
    index_file = os.path.join(INDEX_DIR, f"{course}_index.faiss")
    if not os.path.exists(index_file):
        click.echo(f"Failed to create index for course: {course}")
        sys.exit(1)
    
    if interactive:
        run_interactive_mode(course, show_snippets)
    elif question:
        process_single_query(course, question, show_snippets)
    else:
        click.echo("Error: Provide a question or use --interactive mode")
        sys.exit(1)


def process_single_query(course: str, question: str, show_snippets: bool):
    """Process a single query and display results."""
    try:
        result = query_notes(course, question)
        formatted_result = format_search_result(result, show_snippets)
        
        if CLI_COLORS:
            # Add colors to the output
            formatted_result = add_colors(formatted_result, result.found_in_notes)
        
        click.echo(formatted_result)
        
    except Exception as e:
        click.echo(f"Error processing query: {str(e)}")
        sys.exit(1)


def run_interactive_mode(course: str, show_snippets: bool):
    """Run interactive Q&A session."""
    if CLI_COLORS:
        click.echo(f"{Fore.CYAN}🤖 Notes Q&A Bot - Interactive Mode{Style.RESET_ALL}")
        click.echo(f"{Fore.YELLOW}Course: {course}{Style.RESET_ALL}")
        click.echo(f"{Fore.GREEN}Type 'quit', 'exit', or 'q' to stop{Style.RESET_ALL}")
    else:
        click.echo("🤖 Notes Q&A Bot - Interactive Mode")
        click.echo(f"Course: {course}")
        click.echo("Type 'quit', 'exit', or 'q' to stop")
    
    click.echo("-" * 50)
    
    while True:
        try:
            if CLI_COLORS:
                question = click.prompt(f"{Fore.BLUE}❓ Your question{Style.RESET_ALL}", type=str)
            else:
                question = click.prompt("❓ Your question", type=str)
            
            if question.lower() in ['quit', 'exit', 'q']:
                click.echo("👋 Goodbye!")
                break
            
            if not question.strip():
                continue
            
            click.echo()
            result = query_notes(course, question)
            formatted_result = format_search_result(result, show_snippets)
            
            if CLI_COLORS:
                formatted_result = add_colors(formatted_result, result.found_in_notes)
            
            click.echo(formatted_result)
            click.echo("-" * 50)
            
        except KeyboardInterrupt:
            click.echo("\n👋 Goodbye!")
            break
        except Exception as e:
            click.echo(f"❌ Error: {str(e)}")


def add_colors(text: str, found_in_notes: bool) -> str:
    """Add colors to formatted output."""
    lines = text.split('\n')
    colored_lines = []
    
    for line in lines:
        if line.startswith('Question:'):
            colored_lines.append(f"{Fore.BLUE}{line}{Style.RESET_ALL}")
        elif line.startswith('Answer:'):
            if found_in_notes:
                colored_lines.append(f"{Fore.GREEN}{line}{Style.RESET_ALL}")
            else:
                colored_lines.append(f"{Fore.RED}{line}{Style.RESET_ALL}")
        elif line.startswith('Sources:'):
            colored_lines.append(f"{Fore.YELLOW}{line}{Style.RESET_ALL}")
        elif line.startswith('  •'):
            colored_lines.append(f"{Fore.CYAN}{line}{Style.RESET_ALL}")
        elif line.startswith('Confidence:'):
            colored_lines.append(f"{Fore.MAGENTA}{line}{Style.RESET_ALL}")
        elif line.startswith('Source Snippets:'):
            colored_lines.append(f"{Fore.YELLOW}{line}{Style.RESET_ALL}")
        else:
            colored_lines.append(line)
    
    return '\n'.join(colored_lines)


if __name__ == '__main__':
    main()
