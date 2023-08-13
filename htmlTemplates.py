css = """
    <style>
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    }
    .chat-message.user {
        background-color: #2b313e
    }
    .chat-message.bot {
        background-color: #475063
    }
    .chat-message .avatar {
    width: 20%;
    }
    .chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
    }
    .chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
    }
"""

bot_template = """
    <div class="chat-message bot">
        <div class="avatar">
            <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/ChatGPT_logo.svg/1200px-ChatGPT_logo.svg.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
        </div>
        <div class="message">{{MSG}}</div>
    </div>
"""

user_template = """
    <div class="chat-message user">
        <div class="avatar">
            <img src="https://cdn.theconversation.com/avatars/5289/width238/image-20181128-32191-4rmi38.jpg">
        </div>    
        <div class="message">{{MSG}}</div>
    </div>
"""
