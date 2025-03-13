import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Use direct import since it's in the same directory
from manager import ResearchManager


async def main() -> None:
    query = input("What would you like to research? ")
    await ResearchManager().run(query)


if __name__ == "__main__":
    asyncio.run(main())
