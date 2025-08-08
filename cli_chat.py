#!/usr/bin/env python3
"""
CLI Chat Interface for Clinical RAG System
Provides interactive chat functionality with history support
"""

from RAG_chat_pipeline.core.main import main as initialize_clinical_rag
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


class CLIChatInterface:
    """Command-line interface for chatting with the Clinical RAG system"""

    def __init__(self):
        self.chatbot = None
        self.chat_history = []
        self.session_file = None

    def initialize_rag(self):
        """Initialize the RAG system"""
        print("🚀 Initializing Clinical RAG System...")
        try:
            self.chatbot = initialize_clinical_rag()
            print("✅ Clinical RAG System initialized successfully")
            return True
        except Exception as e:
            print(f"❌ Error initializing Clinical RAG System: {e}")
            return False

    def start_chat_session(self, save_session: bool = True):
        """Start an interactive chat session"""
        if not self.chatbot:
            if not self.initialize_rag():
                return

        if save_session:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_file = f"chat_session_{timestamp}.json"

        print("\n" + "="*60)
        print("🏥 Clinical RAG Chat Interface")
        print("="*60)
        print("Commands:")
        print("  /help     - Show this help message")
        print("  /clear    - Clear chat history")
        print("  /history  - Show chat history")
        print("  /save     - Save session to file")
        print("  /exit     - Exit chat")
        print("="*60)
        print("Start chatting! Ask questions about patient data.\n")

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith('/'):
                    if user_input == '/exit':
                        self._handle_exit()
                        break
                    elif user_input == '/help':
                        self._show_help()
                    elif user_input == '/clear':
                        self._clear_history()
                    elif user_input == '/history':
                        self._show_history()
                    elif user_input == '/save':
                        self._save_session()
                    else:
                        print("Unknown command. Type /help for available commands.")
                    continue

                # Process chat message
                print("Assistant: ", end="", flush=True)

                try:
                    # Use chat method for history-aware responses
                    response = self.chatbot.chat(user_input, self.chat_history)
                    print(response)

                    # Add to chat history
                    self.chat_history.extend([
                        ("user", user_input),
                        ("assistant", response)
                    ])

                    # Auto-save if enabled
                    if save_session and self.session_file:
                        self._save_session(silent=True)

                except Exception as e:
                    print(f"Error: {e}")

            except KeyboardInterrupt:
                print("\n\nChat interrupted by user.")
                self._handle_exit()
                break
            except EOFError:
                print("\n\nChat ended.")
                self._handle_exit()
                break

    def _show_help(self):
        """Show help message"""
        print("\n📋 Available Commands:")
        print("  /help     - Show this help message")
        print("  /clear    - Clear chat history")
        print("  /history  - Show chat history")
        print("  /save     - Save session to file")
        print("  /exit     - Exit chat")
        print("\n💡 Chat Tips:")
        print("  - Ask about specific admission IDs (e.g., 'What diagnoses does admission 12345 have?')")
        print("  - Request specific medical data types (diagnoses, procedures, labs, medications)")
        print("  - Use follow-up questions for more details")
        print("  - The system maintains conversation context")

    def _clear_history(self):
        """Clear chat history"""
        self.chat_history = []
        print("🗑️ Chat history cleared.")

    def _show_history(self):
        """Show chat history"""
        if not self.chat_history:
            print("No chat history yet.")
            return

        print(f"\n📜 Chat History ({len(self.chat_history)//2} exchanges):")
        print("-" * 40)

        for i in range(0, len(self.chat_history), 2):
            if i + 1 < len(self.chat_history):
                user_msg = self.chat_history[i][1]
                assistant_msg = self.chat_history[i + 1][1]

                print(
                    f"You: {user_msg[:100]}{'...' if len(user_msg) > 100 else ''}")
                print(
                    f"Assistant: {assistant_msg[:100]}{'...' if len(assistant_msg) > 100 else ''}")
                print("-" * 40)

    def _save_session(self, silent: bool = False):
        """Save chat session to file"""
        if not self.session_file or not self.chat_history:
            if not silent:
                print("No session to save or no chat history.")
            return

        session_data = {
            "session_file": self.session_file,
            "timestamp": datetime.now().isoformat(),
            "chat_history": self.chat_history,
            "message_count": len(self.chat_history),
            "exchange_count": len(self.chat_history) // 2
        }

        try:
            with open(self.session_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)

            if not silent:
                print(f"💾 Session saved to: {self.session_file}")
        except Exception as e:
            if not silent:
                print(f"Error saving session: {e}")

    def _handle_exit(self):
        """Handle chat exit"""
        if self.chat_history and self.session_file:
            save = input("\nSave this chat session? (y/n): ").lower().strip()
            if save in ['y', 'yes']:
                self._save_session()

        print("👋 Thank you for using Clinical RAG Chat Interface!")

    def load_session(self, session_file: str):
        """Load a previous chat session"""
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                session_data = json.load(f)

            self.chat_history = session_data.get('chat_history', [])
            self.session_file = session_file

            exchange_count = len(self.chat_history) // 2
            print(
                f"📂 Loaded session with {exchange_count} exchanges from: {session_file}")
            return True

        except Exception as e:
            print(f"Error loading session: {e}")
            return False


def main():
    """Main CLI interface"""
    if len(sys.argv) < 2:
        print("""
Clinical RAG CLI Chat Interface

Usage:
    python cli_chat.py <command> [options]

Commands:
    chat                    - Start interactive chat session
    chat --no-save         - Start chat without auto-saving
    load <session_file>    - Load and continue previous session

Examples:
    python cli_chat.py chat
    python cli_chat.py chat --no-save
    python cli_chat.py load chat_session_20250722_143022.json
        """)
        return

    interface = CLIChatInterface()
    command = sys.argv[1].lower()

    if command == "chat":
        save_session = "--no-save" not in sys.argv
        interface.start_chat_session(save_session)

    elif command == "load":
        if len(sys.argv) < 3:
            print("Usage: load <session_file>")
            return

        session_file = sys.argv[2]
        if interface.load_session(session_file):
            interface.start_chat_session(save_session=True)

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
