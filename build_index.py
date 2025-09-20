"""Build index for a course directory."""

import os
import sys
import click
from pathlib import Path

from embeddings import build_course_index
from config import NOTES_DIR


@click.command()
@click.option('--course', '-c', required=True, help='Course name to build index for')
@click.option('--notes-dir', '-d', default=NOTES_DIR, help='Notes directory path')
@click.option('--force', '-f', is_flag=True, help='Force rebuild even if index exists')
def main(course, notes_dir, force):
    """Build FAISS index for a course directory."""
    
    course_dir = os.path.join(notes_dir, course)
    
    if not os.path.exists(course_dir):
        click.echo(f"Error: Course directory not found: {course_dir}")
        click.echo(f"Available courses:")
        if os.path.exists(notes_dir):
            for item in os.listdir(notes_dir):
                if os.path.isdir(os.path.join(notes_dir, item)):
                    click.echo(f"  - {item}")
        sys.exit(1)
    
    # Check if index already exists
    from config import INDEX_DIR
    index_file = os.path.join(INDEX_DIR, f"{course}_index.faiss")
    
    if os.path.exists(index_file) and not force:
        click.echo(f"Index already exists for {course}. Use --force to rebuild.")
        sys.exit(0)
    
    try:
        click.echo(f"Building index for course: {course}")
        click.echo(f"Processing directory: {course_dir}")
        click.echo("-" * 50)
        
        index_path = build_course_index(course, course_dir)
        
        click.echo("-" * 50)
        click.echo(f"Index built successfully!")
        click.echo(f"Index saved to: {index_path}")
        click.echo(f"\nYou can now query this course using:")
        click.echo(f"  python query.py --course {course} \"Your question here\"")
        
    except Exception as e:
        click.echo(f"Error building index: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
