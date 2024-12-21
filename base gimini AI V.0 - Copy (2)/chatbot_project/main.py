import sys
import os
import json
import datetime
from typing import Dict, List, Any, Optional

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLineEdit, QPushButton, QListWidget
from PyQt6.QtGui import QColor, QPalette, QFont
from PyQt6.QtCore import Qt, QPropertyAnimation, QEasingCurve

import google.generativeai as genai
from dotenv import load_dotenv

import core_functions
import prompt_utils

load_dotenv()  # Moved to the beginning
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


BRAIN_MEMORY_FOLDER = "brain_memory"
MEMORY_FILE = "memory.json"

KNOWLEDGE_BASE = {
    "general": {
        "AI": "Artificial Intelligence",
        "ML": "Machine Learning",
        "DL": "Deep Learning"
    },
    "functions": {
        "save": "Save conversation history",
        "clear": "Clear conversation history"
    },
    "programming_languages": {
        "Python": ["Easy to learn", "High-level language"],
        "Java": ["Object-oriented", "Platform independent"],
        "C++": ["High-performance", "Compiled language"]
    },
    "ai_models": {
        "Gemini": ["Conversational AI", "Language model"],
        "BERT": ["Language model", "Pre-trained"],
        "RoBERTa": ["Language model", "Pre-trained"]
    },
    "commonsense": {
        "birds_can_fly": "Birds are typically able to fly.",
        "water_is_wet": "Water is typically considered wet.",
        "fire_is_hot": "Fire is typically considered hot.",
        "sky_is_blue": "The sky is often blue during the day",
        "sun_is_hot": "The sun is typically considered to be hot",
        "ice_is_cold": "Ice is typically considered cold"
    }
}


def initialize_memory_folder():
    """Creates the brain memory folder if it doesn't exist."""
    if not os.path.exists(BRAIN_MEMORY_FOLDER):
        os.makedirs(BRAIN_MEMORY_FOLDER)
        print("'{}' folder created.".format(BRAIN_MEMORY_FOLDER))
    else:
        print("'{}' folder exists.".format(BRAIN_MEMORY_FOLDER))


def generate_suggestions(user_message):
    """Generates dynamic suggestions based on the user input."""
    user_message = user_message.lower()
    suggestions = []

    if "hello" in user_message or "hi" in user_message or "hey" in user_message or "yo" in user_message:
        suggestions = ["How can you help?", "What can you do?", "Tell me about yourself?", "What is your memory?",
                       "How do you work?"]
    elif "how are you" in user_message:
        suggestions = ["What is your main purpose?", "What do you know?", "Can you tell me a joke?",
                       "Do you remember anything?", "What is your personality?"]
    elif "weather" in user_message:
        suggestions = ["What's the time?", "How do you work?", "Who created you?", "How is your memory?",
                       "What is your best feature?"]
    elif "bye" in user_message or "goodbye" in user_message:
        suggestions = ["Goodbye", "See you later", "Have a great day"]
    elif "save" not in user_message:
        suggestions = ["Tell me more", "Give me an example", "Explain this further", "How does your memory work?",
                       "Do you retain information?"]

    return suggestions


def analyze_intent(message):
    """Classifies the intent of a message into general categories."""
    message = message.lower()
    if "hello" in message or "hi" in message or "hey" in message or "yo" in message:
        return "greeting"
    elif "how are you" in message:
        return "question"
    elif "what" in message or "where" in message or "when" in message or "who" in message or "why" in message:
        return "question"
    elif "save" in message:
        return "command"
    else:
        return "declaration"


def extract_entities(message):
    """Extracts entities from the given message."""
    entities = {}
    # Very basic entity extraction
    if "gemini" in message.lower():
        entities["ai_model"] = "gemini"
    if "python" in message.lower():
        entities["programming_language"] = "python"
    # Add further pattern-based extraction here.
    return entities


def logical_inference(message, entities):
    """Performs logical inference based on user input and extracted entities."""
    inferred_knowledge = {}

    if "like" in message.lower() and "programming_language" in entities:
        lang = entities["programming_language"]
        inferred_knowledge["user_preference"] = "User likes programming language {}".format(lang)

    if "favorite" in message.lower() and "ai_model" in entities:
        model = entities["ai_model"]
        inferred_knowledge["user_favorite"] = "User's favorite AI model is {}".format(model)

    return inferred_knowledge


class ChatbotUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Chatbot")
        self.setMinimumSize(800, 600)

        # Set up the main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        self.setCentralWidget(main_widget)

        # Chat display
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        main_layout.addWidget(self.chat_display)

        # User input area
        input_layout = QHBoxLayout()
        self.user_input = QLineEdit()
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        input_layout.addWidget(self.user_input)
        input_layout.addWidget(self.send_button)
        main_layout.addLayout(input_layout)

        # Suggestions area
        self.suggestions = QListWidget()
        self.suggestions.itemClicked.connect(self.use_suggestion)
        main_layout.addWidget(self.suggestions)

        # Apply styles
        self.apply_styles()

        self.conversation_history = []
        self.memory = core_functions.retrieve_memory_from_file(os.path.join(BRAIN_MEMORY_FOLDER, MEMORY_FILE))

    def apply_styles(self):
        # Set dark theme palette
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
        QApplication.instance().setPalette(palette)

        # Set font
        font = QFont("Roboto", 10)
        QApplication.instance().setFont(font)

        # Apply stylesheets for modern look
        style = """
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
            border-radius: 5px;
        }
        QTextEdit, QLineEdit {
            background-color: #3b3b3b;
            border: 1px solid #555555;
            padding: 5px;
        }
        QPushButton {
            background-color: #0d6efd;
            color: white;
            border: none;
            padding: 8px 16px;
        }
        QPushButton:hover {
            background-color: #0b5ed7;
        }
        QListWidget {
            background-color: #3b3b3b;
            border: 1px solid #555555;
        }
        QListWidget::item:hover {
            background-color: #4b4b4b;
        }
        """
        self.setStyleSheet(style)


    def send_message(self):
        message = self.user_input.text().strip()
        if not message:
            self.receive_message("AI: Please enter a message.")
            return
        self.chat_display.append(f"You: {message}")
        self.user_input.clear()
        intent = analyze_intent(message)
        entities = extract_entities(message)
        print("Intent: {}".format(intent))
        print("Entities: {}".format(entities))

        try:
            knowledge_results = core_functions.search_knowledge_base(KNOWLEDGE_BASE, message, message) # needs to be case-insensitive.
            formatted_conversation = [item["message"] for item in self.memory.get("conversation_history", [])] if self.memory else []
            response = prompt_utils.generate_response(message,formatted_conversation, self.memory, knowledge_results)
            if response:
                self.receive_message(f"AI: {response}")
                inferred_knowledge = logical_inference(message, entities)
                if inferred_knowledge:
                    self.receive_message("AI: Inferred knowledge:\n{}".format('\n'.join(inferred_knowledge.values())))
                self.memory = core_functions.update_memory(self.memory, message, response, intent, entities)
                core_functions.save_memory(self.memory, os.path.join(BRAIN_MEMORY_FOLDER, MEMORY_FILE))
            else:
                self.receive_message("AI: Sorry, I encountered an error processing your request.")

        except Exception as e:
            print(f"Error in main loop: {e}")
            self.receive_message("AI: Sorry, there was an unexpected error.")
        self.update_suggestions(message)

    def receive_message(self, message):
         # Animate the new message appearance
         original_height = self.chat_display.height()
         self.chat_display.append(message)
         new_height = self.chat_display.document().size().height()

         animation = QPropertyAnimation(self.chat_display, b"maximumHeight")
         animation.setDuration(300)
         animation.setStartValue(original_height)
         animation.setEndValue(new_height)
         animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
         animation.start()

    def update_suggestions(self, user_message):
        suggestions = generate_suggestions(user_message)
        self.suggestions.clear()
        self.suggestions.addItems(suggestions)

    def use_suggestion(self, item):
        self.user_input.setText(item.text())


if __name__ == "__main__":
    initialize_memory_folder()
    try:
        app = QApplication(sys.argv)
        window = ChatbotUI()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        print(f"Unhandled exception: {e}")
    finally:
       print("Exiting application")