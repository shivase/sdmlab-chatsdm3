import os
from slack_bolt import App
from dotenv import load_dotenv

from Agents import Agents

load_dotenv()

agents = Agents()

def process_event(event, say, memory_key):
    text: str = event['text']

    if text.startswith('initial'):
        agents.set_prompt(text)
        say(text='system定義を設定しました')
    elif text == 'reset':
        agents.delete(memory_key)
        say(text='会話をリセットしました')
    else:
        agent = agents.get(memory_key)
        res = agent.run(input=text)

        say(text=res)

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
