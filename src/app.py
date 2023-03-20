import os
from slack_bolt import App
from dotenv import load_dotenv
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

from Memories import Memories
from Prompt import Prompt

load_dotenv()

model = OpenAI(open_ai_api_key=os.environ['OPEN_API_KEY'], temperature=0.9)

prompt = Prompt()
memories = Memories()

def process_event(event, say, memory_key):
    text = event['text']

    if text.startswith('initial'):
        prompt.set(text)
        say(text='system定義を設定しました')
    else:
        memory = memories.get_memory(memory_key)
        chain = ConversationChain(memory=memory, prompt=prompt.get(), llm=model)
        res = chain.call(input=text)

        say(text=res.response)

        if len(memory.chat_history.messages) > 20:
            memories.delete_memory(memory_key)
            say(text='(system) 会話制限（20回)に到達したため、履歴は削除されました')


if __name__ == "__main__":

    slack = App(
        token=os.environ.get("SLACK_BOT_TOKEN"),
        signing_secret=os.environ.get("SLACK_SIGNING_SECRET")
    )

    @slack.event("message")
    def handle_message(body, say):
        event = body['event']
        process_event(event, say, event['user'])

    @slack.event("app_mention")
    def handle_app_mention(body, say):
        event = body['event']
        thread_ts = event.get('thread_ts', event['ts'])
        process_event(event, say, thread_ts)

    slack.start(port=int(os.environ.get("PORT")))
