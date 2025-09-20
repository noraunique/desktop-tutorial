"""
GUI Application for Notes Q&A Chatbot
A user-friendly desktop interface for querying your notes.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, colorchooser
import threading
from datetime import datetime
import os
import subprocess
import pickle
import json
import sys

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrieval import query_notes, studybot_query, studybot_query_with_meta
from config import NOTES_DIR, INDEX_DIR, SUPPORTED_EXTENSIONS

def check_and_rebuild_index(course):
    """Check if index needs rebuilding and rebuild if necessary."""
    index_file = os.path.join(INDEX_DIR, f"{course}_index.faiss")
    metadata_file = os.path.join(INDEX_DIR, f"{course}_metadata.pkl")
    
    # Check if index exists
    if not os.path.exists(index_file) or not os.path.exists(metadata_file):
        print(f"Index not found for {course}. Building...")
        rebuild_index(course)
        return
    
    # Check if any files are newer than the index
    course_dir = os.path.join(NOTES_DIR, course)
    if not os.path.exists(course_dir):
        return
    
    index_mtime = os.path.getmtime(index_file)
    
    for root, dirs, files in os.walk(course_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                file_path = os.path.join(root, file)
                if os.path.getmtime(file_path) > index_mtime:
                    print(f"Found newer file: {file}. Rebuilding index...")
                    rebuild_index(course)
                    return

def rebuild_index(course):
    """Rebuild the index for a course."""
    try:
        result = subprocess.run([
            sys.executable, "build_index.py", 
            "--course", course, "--force"
        ], capture_output=True, text=True, cwd=os.getcwd())
        
        if result.returncode != 0:
            raise Exception(f"Build failed: {result.stderr}")
    except Exception as e:
        raise Exception(f"Error rebuilding index: {str(e)}")

class NotesQAChatbot:
    def __init__(self, root):
        self.root = root
        self.root.title("📚 Notes Q&A Chatbot")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        # Load settings
        self.settings = self.load_settings()
        self.user_name = self.settings.get("user_name", "Delsan")
        self.bot_name = self.settings.get("bot_name", "Diloxcya")
        self.theme = self.settings.get("theme", "default")
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.apply_theme()
        
        self.course = "DSA"  # Default course - set before setup_ui
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title with settings button
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 20))
        title_frame.columnconfigure(1, weight=1)
        
        title_label = ttk.Label(title_frame, text="📚 Notes Q&A Chatbot", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, sticky=(tk.W))
        
        settings_btn = ttk.Button(title_frame, text="⚙️ Settings", 
                                 command=self.open_settings, width=12)
        settings_btn.grid(row=0, column=2, sticky=(tk.E))
        
        # Course and file selection
        course_frame = ttk.Frame(main_frame)
        course_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        course_frame.columnconfigure(3, weight=1)
        
        # Course selection
        ttk.Label(course_frame, text="Course:").grid(row=0, column=0, padx=(0, 10))
        self.course_var = tk.StringVar(value="DSA")
        course_combo = ttk.Combobox(course_frame, textvariable=self.course_var, 
                                   values=["DSA"], state="readonly", width=12)
        course_combo.grid(row=0, column=1, padx=(0, 20))
        course_combo.bind('<<ComboboxSelected>>', self.on_course_change)
        
        # File selection dropdown
        ttk.Label(course_frame, text="Files:").grid(row=0, column=2, padx=(0, 10))
        self.selected_file_var = tk.StringVar(value="All Files")
        self.file_combo = ttk.Combobox(course_frame, textvariable=self.selected_file_var, 
                                      state="readonly", width=25)
        self.file_combo.grid(row=0, column=3, padx=(0, 10), sticky=(tk.W, tk.E))
        
        # Refresh button
        refresh_btn = ttk.Button(course_frame, text="🔄 Refresh", 
                                command=self.refresh_files, width=10)
        refresh_btn.grid(row=0, column=4, padx=(0, 20))
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(course_frame, textvariable=self.status_var, 
                                foreground="green")
        status_label.grid(row=0, column=5)
        
        # Chat area (glass card with subtle shadow and float)
        chat_outer = ttk.Frame(main_frame, style='Shadow.TFrame')
        chat_outer.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 14))
        chat_outer.columnconfigure(0, weight=1)
        chat_outer.rowconfigure(0, weight=1)

        chat_frame = ttk.Frame(chat_outer, padding="8 8 8 8", style='Glass.TFrame')
        chat_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        # Chat display
        self.chat_display = scrolledtext.ScrolledText(
            chat_frame, 
            wrap=tk.WORD, 
            width=80, 
            height=20,
            font=('Consolas', 10),
            state=tk.DISABLED
        )
        self.chat_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Input area (glass card)
        input_outer = ttk.Frame(main_frame, style='Shadow.TFrame')
        input_outer.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(6, 0))
        input_outer.columnconfigure(0, weight=1)

        input_frame = ttk.Frame(input_outer, padding="8 8 8 8", style='Glass.TFrame')
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        input_frame.columnconfigure(0, weight=1)
        
        # Question input
        self.question_var = tk.StringVar()
        self.question_entry = ttk.Entry(input_frame, textvariable=self.question_var, 
                                       font=('Arial', 11))
        self.question_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.question_entry.bind('<Return>', self.on_ask_question)
        
        # Language selector (StudyBot)
        ttk.Label(input_frame, text="Language:").grid(row=0, column=1, padx=(0, 8))
        self.lang_var = tk.StringVar(value="English")
        self.lang_combo = ttk.Combobox(
            input_frame,
            textvariable=self.lang_var,
            values=["English", "Tamil", "Sinhala"],
            state="readonly",
            width=12
        )
        self.lang_combo.grid(row=0, column=2, padx=(0, 10))

        # Ask button
        self.ask_button = ttk.Button(input_frame, text="Ask Question",
                                     command=self.on_ask_question, width=15, style='Accent.TButton')
        self.ask_button.grid(row=0, column=3, padx=(0, 10))
        
        # Clear button
        clear_button = ttk.Button(input_frame, text="Clear Chat",
                                  command=self.clear_chat, width=12)
        clear_button.grid(row=0, column=4)
        
        # Initial welcome message
        welcome_msg = ("Welcome to Notes Q&A Chatbot!\n" +
                      "Ask questions about your notes and get instant answers with sources.\n" +
                      "The system will automatically rebuild the index when new files are detected.\n" +
                      "Current course: DSA\n" + "="*60)
        self.add_message("System", welcome_msg)

        # Start subtle floating animation on glass cards
        self._float_phase = 0
        self._float_widgets = [chat_outer, input_outer]
        self._animate_float()
        
        # Focus on input
        self.question_entry.focus()
        
        # Load and display available files
        self.update_files_display()
    
    def update_files_display(self):
        """Update the files dropdown with current indexed files."""
        try:
            course = self.course_var.get()
            import pickle
            metadata_file = os.path.join(INDEX_DIR, f"{course}_metadata.pkl")
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    chunks = data['chunks']
                    
                # Get unique filenames
                filenames = sorted(set(chunk.filename for chunk in chunks))
                
                # Update dropdown with all files
                file_options = ["All Files"] + filenames
                self.file_combo['values'] = file_options
                
                # Set default selection
                if self.selected_file_var.get() not in file_options:
                    self.selected_file_var.set("All Files")
                    
            else:
                self.file_combo['values'] = ["No files indexed yet"]
                self.selected_file_var.set("No files indexed yet")
                
        except Exception as e:
            self.file_combo['values'] = ["Error loading files"]
            self.selected_file_var.set("Error loading files")
    
    def on_course_change(self, event=None):
        """Handle course selection change."""
        self.update_files_display()
    
    def refresh_files(self):
        """Manually refresh files and rebuild index if needed."""
        def refresh_thread():
            try:
                course = self.course_var.get()
                
                # Update status
                self.root.after(0, lambda: self.status_var.set("Scanning for new files..."))
                self.root.after(0, lambda: self.add_message("System", "🔄 Scanning for new files..."))
                
                # Force rebuild index to detect new files
                self.root.after(0, lambda: self.status_var.set("Rebuilding index..."))
                rebuild_index(course)
                
                # Update files display
                self.root.after(0, self.update_files_display)
                
                # Success message
                self.root.after(0, lambda: self.status_var.set("Files refreshed!"))
                self.root.after(0, lambda: self.add_message("System", "✅ Files refreshed! Index updated with any new files."))
                
                # Reset status after 3 seconds
                self.root.after(3000, lambda: self.status_var.set("Ready"))
                
            except Exception as e:
                error_msg = f"Error refreshing files: {str(e)}"
                self.root.after(0, lambda: self.status_var.set("Refresh failed"))
                self.root.after(0, lambda: self.add_message("System", f"❌ {error_msg}"))
                self.root.after(3000, lambda: self.status_var.set("Ready"))
        
        # Run in separate thread to avoid blocking GUI
        import threading
        thread = threading.Thread(target=refresh_thread, daemon=True)
        thread.start()
    
    def add_message(self, sender, message, color="black"):
        """Add a message to the chat display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Enable editing temporarily
        self.chat_display.config(state=tk.NORMAL)
        
        # Add timestamp and sender
        if sender == "System":
            self.chat_display.insert(tk.END, f"[{timestamp}] {sender}: ", "system")
        elif sender == self.user_name:
            self.chat_display.insert(tk.END, f"[{timestamp}] {sender}: ", "user")
        else:
            self.chat_display.insert(tk.END, f"[{timestamp}] {sender}: ", "bot")
        
        # Add message
        self.chat_display.insert(tk.END, f"{message}\n\n")
        
        # Configure tags for colors
        self.chat_display.tag_config("system", foreground="blue", font=('Consolas', 10, 'bold'))
        self.chat_display.tag_config("user", foreground="green", font=('Consolas', 10, 'bold'))
        self.chat_display.tag_config("bot", foreground="purple", font=('Consolas', 10, 'bold'))
        
        # Disable editing
        self.chat_display.config(state=tk.DISABLED)
        
        # Auto-scroll to bottom
        self.chat_display.see(tk.END)
    
    def clear_chat(self):
        """Clear the chat display."""
        self.chat_display.config(state=tk.NORMAL)
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state=tk.DISABLED)
        
        # Add welcome message again
        self.add_message("System", "Chat cleared. Ready for new questions!")
    
    def on_ask_question(self, event=None):
        """Handle question submission."""
        question = self.question_var.get().strip()
        if not question:
            return
        
        # Clear input
        self.question_var.set("")
        
        # Add user question to chat
        self.add_message(self.user_name, question)
        
        # Disable button and show processing
        self.ask_button.config(state="disabled")
        self.status_var.set("Processing...")
        self.root.update()
        
        # Process question in separate thread to avoid freezing UI
        thread = threading.Thread(target=self.process_question, args=(question,))
        thread.daemon = True
        thread.start()
    
    def process_question(self, question):
        """Process the question and get answer."""
        try:
            course = self.course_var.get()
            
            # Update status
            self.root.after(0, lambda: self.status_var.set("Checking for new files..."))
            
            # Check and rebuild index if needed
            check_and_rebuild_index(course)
            
            # Update files display after potential rebuild
            self.root.after(0, self.update_files_display)
            
            # Update status
            self.root.after(0, lambda: self.status_var.set("Searching for answer..."))
            
            # Determine TARGET_LANG from dropdown
            lang_map = {"English": "en", "Tamil": "ta", "Sinhala": "si"}
            target_lang = lang_map.get(self.lang_var.get(), "en")

            # Use StudyBot formatter and capture top score for debug/telemetry
            response, top_score = studybot_query_with_meta(course, question, target_lang)
            
            # Add bot response to chat
            self.root.after(0, lambda: self.add_message(self.bot_name, response))
            # Show score in status bar briefly
            self.root.after(0, lambda: self.status_var.set(f"Top score: {top_score:.2f}"))
            self.root.after(3000, lambda: self.status_var.set("Ready"))
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            self.root.after(0, lambda: self.add_message("System", error_msg))
        
        finally:
            # Re-enable button and update status
            self.root.after(0, lambda: self.ask_button.config(state="normal"))
            self.root.after(0, lambda: self.status_var.set("Ready"))
    
    def load_settings(self):
        """Load user settings from file."""
        settings_file = "chatbot_settings.json"
        try:
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}
    
    def save_settings(self):
        """Save user settings to file."""
        settings_file = "chatbot_settings.json"
        try:
            with open(settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception:
            pass
    
    def apply_theme(self):
        """Apply the selected theme colors."""
        # Base palette
        bg = '#0f172a'  # slate-900
        fg = '#e5e7eb'  # gray-200
        glass_bg = '#1f2937'  # slate-800
        shadow = '#0b1224'
        accent_from = '#4c82ff'
        accent_to = '#7c5cff'

        self.root.configure(bg=bg)
        # Labels / frames
        self.style.configure('TLabel', background=bg, foreground=fg)
        self.style.configure('TFrame', background=bg)
        # Glass card style
        self.style.configure('Glass.TFrame', background=glass_bg, relief='flat')
        self.style.configure('Shadow.TFrame', background=shadow)
        # Buttons
        self.style.configure('TButton', background='#334155', foreground=fg, padding=6)
        self.style.configure('Accent.TButton', background='#3b82f6', foreground='white', padding=8)
        # Entry/Combobox colors (not all themes honor these fully)
        self.style.map('Accent.TButton', relief=[('pressed','sunken'), ('active','raised')])
    
    def open_settings(self):
        """Open the settings dialog."""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("⚙️ Settings")
        settings_window.geometry("400x350")
        settings_window.resizable(False, False)
        settings_window.grab_set()  # Make it modal
        
        # Center the window
        settings_window.transient(self.root)
        
        main_frame = ttk.Frame(settings_window, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        settings_window.columnconfigure(0, weight=1)
        settings_window.rowconfigure(0, weight=1)
        
        # Names section
        names_frame = ttk.LabelFrame(main_frame, text="Display Names", padding="10")
        names_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        names_frame.columnconfigure(1, weight=1)
        
        ttk.Label(names_frame, text="Your Name:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        user_name_var = tk.StringVar(value=self.user_name)
        user_entry = ttk.Entry(names_frame, textvariable=user_name_var, width=20)
        user_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=(0, 10))
        
        ttk.Label(names_frame, text="Bot Name:").grid(row=1, column=0, sticky=tk.W, padx=(0, 10))
        bot_name_var = tk.StringVar(value=self.bot_name)
        bot_entry = ttk.Entry(names_frame, textvariable=bot_name_var, width=20)
        bot_entry.grid(row=1, column=1, sticky=(tk.W, tk.E))
        
        # Theme section
        theme_frame = ttk.LabelFrame(main_frame, text="Interface Theme", padding="10")
        theme_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        theme_var = tk.StringVar(value=self.theme)
        ttk.Radiobutton(theme_frame, text="Default", variable=theme_var, value="default").grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(theme_frame, text="Light", variable=theme_var, value="light").grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(theme_frame, text="Dark", variable=theme_var, value="dark").grid(row=2, column=0, sticky=tk.W)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(15, 0))
        button_frame.columnconfigure(0, weight=1)
        
        def save_and_close():
            # Update settings
            self.user_name = user_name_var.get().strip() or "Delsan"
            self.bot_name = bot_name_var.get().strip() or "Diloxcya"
            self.theme = theme_var.get()
            
            # Save to settings dict
            self.settings["user_name"] = self.user_name
            self.settings["bot_name"] = self.bot_name
            self.settings["theme"] = self.theme
            
            # Apply theme
            self.apply_theme()
            
            # Save to file
            self.save_settings()
            
            # Show confirmation
            self.add_message("System", f"Settings updated! User: {self.user_name}, Bot: {self.bot_name}, Theme: {self.theme}")
            
            settings_window.destroy()
        
        ttk.Button(button_frame, text="Save & Close", command=save_and_close).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(button_frame, text="Cancel", command=settings_window.destroy).grid(row=0, column=1)

    def _animate_float(self):
        """Subtle floating effect by toggling padding on outer glass frames."""
        try:
            self._float_phase = (self._float_phase + 1) % 60  # 60-step cycle
            dy = -2 if self._float_phase < 30 else 0
            for w in getattr(self, '_float_widgets', []):
                info = w.grid_info()
                pady = info.get('pady', (0, 0))
                if isinstance(pady, tuple):
                    base_top, base_bot = 0, 14 if info.get('row', 0) == 2 else 6
                else:
                    base_top, base_bot = 0, int(pady)
                new_top = base_top + dy
                new_bot = base_bot - dy
                try:
                    w.grid_configure(pady=(new_top, new_bot))
                except Exception:
                    pass
        finally:
            self.root.after(200, self._animate_float)

def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = NotesQAChatbot(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        root.quit()

if __name__ == "__main__":
    main()
