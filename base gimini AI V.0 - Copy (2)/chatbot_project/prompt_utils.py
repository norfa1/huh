import jax
import jax.numpy as jnp
import google.generativeai as genai
import os
from typing import List, Dict, Any, Optional


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)


def _build_prompt(prompt: str, memory_context: str, feedback_context: str,
                 knowledge_results: str, formatted_conversation: str) -> str:
    """Builds the final prompt for the LLM.

    Args:
        prompt (str): The user's input prompt.
        memory_context (str): Context from memory.
        feedback_context (str): Feedback from the user.
        knowledge_results (str): Relevant knowledge base results.
        formatted_conversation (str): Formatted conversation history.

    Returns:
        str: The complete prompt.
    """
    system_prompt = """You are Gemini, a helpful and engaging AI assistant. Your responses should be:
    
        1. Direct and relevant to the user's question
        2. Contextually aware of the conversation history
        3. Natural and conversational, not repetitive
        4. Properly formatted without echoing back the user's messages
        
        If the user's message is casual (like "yo" or "hey"), respond naturally with a friendly greeting and ask how you can help.
        Never repeat the user's message back to them unless specifically asked to do so.
        """

    modified_prompt = f"""{system_prompt}
    
        {formatted_conversation}
    
        Here is some information to help you respond:
        {memory_context}
    
        {feedback_context}
    
        {knowledge_results}
    
        Current user message: {prompt}
        
        Respond naturally and appropriately to the user's message."""
    return modified_prompt


def generate_response(prompt: str, conversation_history: List[str], memory: Dict[str, Any],
                      knowledge_results: str, feedback: Optional[str] = None,
                      model_name: str = "gemini-pro") -> str:
    """Generates a text response using the Gemini Pro model.

    Args:
        prompt (str): The user's input prompt.
        conversation_history (list): List of previous prompts and responses.
         memory (dict): Memory dictionary.
        knowledge_results (str): Relevant knowledge from the knowledge base.
        feedback (str, optional): Feedback from the user. Defaults to None.
        model_name (str, optional): Name of the Gemini model. Defaults to "gemini-pro".

    Returns:
        str: The generated text response.
    """
    if not prompt:
        return "Prompt cannot be empty."

    memory_context = ""
    if memory:
        user_preferences = memory.get("user_preferences", {})
        project_details = memory.get("project_details", {})
        personal_information = memory.get("personal_information", {})

        if user_preferences:
            memory_context += "User Preferences: {}\n".format(user_preferences)
        if project_details:
            memory_context += "Project Details: {}\n".format(project_details)
        if personal_information:
            memory_context += "Personal Information: {}\n".format(personal_information)

    formatted_conversation = ""
    if conversation_history:
        formatted_conversation += "\nConversation History:\n"
        for turn in conversation_history:
            if turn:
                formatted_conversation += "{}\n".format(turn)

    if feedback:
        feedback_context = "You have received feedback: {0}. You must change how you respond to be more helpful".format(
            feedback)
    else:
        feedback_context = ""

    modified_prompt = _build_prompt(prompt, memory_context, feedback_context,
                                    knowledge_results, formatted_conversation)
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(modified_prompt)
        return response.text
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Sorry, I encountered an issue. Please try again."