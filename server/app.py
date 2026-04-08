import asyncio
from inference import main as run_inference


def main():
    """
    Entry point required by OpenEnv validator.
    Must be synchronous wrapper.
    """
    asyncio.run(run_inference())


# REQUIRED for validator
if __name__ == "__main__":
    main()
