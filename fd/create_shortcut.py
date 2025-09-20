import os
import winshell
from win32com.client import Dispatch

def create_desktop_shortcut():
    """Create a desktop shortcut for the Notes Q&A Chatbot."""
    
    # Get desktop path
    desktop = winshell.desktop()
    
    # Path to the batch file
    batch_file = os.path.join(os.getcwd(), "run_chatbot.bat")
    
    # Create shortcut path
    shortcut_path = os.path.join(desktop, "Notes Q&A Chatbot.lnk")
    
    # Create shortcut
    shell = Dispatch('WScript.Shell')
    shortcut = shell.CreateShortCut(shortcut_path)
    shortcut.Targetpath = batch_file
    shortcut.WorkingDirectory = os.getcwd()
    shortcut.IconLocation = batch_file
    shortcut.Description = "Notes Q&A Chatbot - AI-powered document search"
    shortcut.save()
    
    print(f"Desktop shortcut created: {shortcut_path}")
    return shortcut_path

if __name__ == "__main__":
    try:
        create_desktop_shortcut()
        print("✅ Desktop shortcut created successfully!")
        print("You can now double-click 'Notes Q&A Chatbot' on your desktop to run the application.")
    except Exception as e:
        print(f"❌ Error creating shortcut: {e}")
        print("You can manually create a shortcut by:")
        print("1. Right-click on desktop → New → Shortcut")
        print("2. Browse to: d:\\fd\\run_chatbot.bat")
        print("3. Name it: Notes Q&A Chatbot")
