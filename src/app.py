import os
from slack_bolt import App
from dotenv import load_dotenv

from conversation_agent import ConversationAgent

load_dotenv()
agent = ConversationAgent()

def process_event(event, say, memory_key):
    text: str = event['text']
    thread_ts = event.get("thread_ts", None) or event["ts"]

    if text.startswith('initial'):
        agent.set_prefix(text)
        say(text='system定義を設定しました',  thread_ts=thread_ts)
    elif text == 'reset':
        agent.delete(memory_key)
        say(text='会話をリセットしました',  thread_ts=thread_ts)
    else:
        executor = agent.get_executor(memory_key)
        res = executor.run(input=text)
        say(text=res, thread_ts=thread_ts)

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