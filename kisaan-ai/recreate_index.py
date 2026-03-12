import os
import time
from pinecone import Pinecone, ServerlessSpec

PINECONE_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_KEY:
    print("ERROR: PINECONE_API_KEY not set. Run: export $(cat .env | grep -v '^#' | xargs)")
    exit(1)

pc = Pinecone(api_key=PINECONE_KEY)

print("Deleting old index 'kisaan-ai'...")
pc.delete_index("kisaan-ai")
print("Deleted ✓")

print("Waiting 10 seconds...")
time.sleep(10)

print("Creating new index at 384-dim...")
pc.create_index(
    name="kisaan-ai",
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)

print("Waiting for index to be ready...")
while not pc.describe_index("kisaan-ai").status["ready"]:
    print("  still initializing...")
    time.sleep(3)

print("Index 'kisaan-ai' ready at 384-dim ✓")
