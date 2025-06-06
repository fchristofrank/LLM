#!/usr/bin/env python3
import itertools
import hashlib
import sys
import time

# Read the file
filename = sys.argv[1] if len(sys.argv) > 1 else "franklin.ch.MD5-04"
with open(filename, 'r') as f:
    lines = f.readlines()
    message_line = lines[0].strip()
    hash_line = lines[1].strip()

# Extract just the hex message and hash
message_with_x = message_line.split(': ')[1]
target_hash = hash_line.split(': ')[1]

print(f"Original Message: {message_with_x}")
print(f"MD5 Hash: {target_hash}")

test_message = "d59821dd66f047d30000fff95fa689a5"
test_full = f"cy5001s25{test_message}"  # NO $ sign!
test_hash = hashlib.md5(test_full.encode()).hexdigest()
print(f"\nTest with professor's example:")
print(f"Message: {test_full}")
print(f"Hash: {test_hash}")
print(f"Expected: 8b66557c7de44e0fa8cf341350ff0157")
print(f"Match: {test_hash == '8b66557c7de44e0fa8cf341350ff0157'}")

# Split the message at X's
parts = message_with_x.split('X')
num_x = len(parts) - 1

print(f"\nNumber of X's to replace: {num_x}")
print(f"Starting brute force...")

# Brute force
attempts = 0
start_time = time.time()

for attempt in itertools.product('0123456789abcdef', repeat=num_x):
    # Build message
    message = parts[0]
    for i, char in enumerate(attempt):
        message += char + parts[i+1]

    # Calculate MD5 - NO $ sign between cy5001s25 and message!
    full_message = f"cy5001s25{message}"
    md5_hash = hashlib.md5(full_message.encode()).hexdigest()

    attempts += 1
    if attempts % 1000000 == 0:
        elapsed = time.time() - start_time
        rate = attempts / elapsed
        print(f"Progress: {attempts:,} attempts, {rate:,.0f} attempts/sec, {elapsed:.1f}s")

    if md5_hash == target_hash:
        elapsed = time.time() - start_time
        print(f"\nMatch found: {message}")
        print(f"MD5 Hash: {md5_hash}")
        print(f"Time taken: {elapsed:.2f} seconds")
        print(f"Attempts: {attempts:,}")

        # Save to file
        with open('franklin.ch.p22.md5.flag', 'w') as f:
            f.write(message + '\n')

        break
else:
    elapsed = time.time() - start_time
    print(f"\nNo match found after {attempts:,} attempts in {elapsed:.2f} seconds")