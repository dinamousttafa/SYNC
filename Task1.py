import nltk
from nltk.chat.util import Chat, reflections

chatbot_responses = [
    (r'hi|hello|hey', ['Hello!', 'Hi there!', 'Hey!']),
    (r'how are you?', ['I am doing well, thank you!', 'I am fine, thanks for asking.']),
    (r'what is your name?', ['My name is ChatBot.', 'I am called ChatBot.']),
    (r'bye|goodbye', ['Goodbye!', 'See you later!', 'Bye!']),
]

chatbot = Chat(chatbot_responses, reflections)


print("ChatBot: Hello, I am ChatBot. How can I help you today? (type 'quit' to exit)")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    response = chatbot.respond(user_input)
    print("ChatBot:", response)
