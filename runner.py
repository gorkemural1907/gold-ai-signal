import time
import traceback
import signal_bot

SLEEP_SECONDS = 60

def main():
    while True:
        try:
            signal_bot.main()
        except Exception:
            traceback.print_exc()
        time.sleep(SLEEP_SECONDS)

if __name__ == "__main__":
    main()
