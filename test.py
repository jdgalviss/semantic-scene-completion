import concurrent.futures
import time

def print_numbers():
    for i in range(10):
        print(i)
        time.sleep(1)

def print_letters():
    for letter in 'abcdefghij':
        print(letter)
        time.sleep(1)


# Create ThreadPoolExecutor
executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

# Submit tasks to executor
future1 = executor.submit(print_numbers)
future2 = executor.submit(print_letters)

# Wait for both tasks to finish
concurrent.futures.wait([future1, future2])

print("Finished")