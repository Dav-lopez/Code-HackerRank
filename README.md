# Code-HackerRank
Question 1 
![image](https://github.com/user-attachments/assets/7433ebb5-a1ea-480f-ae69-1b55f790a306)
![image](https://github.com/user-attachments/assets/62ce7161-b888-4ece-9e14-c2a2a585ce94)
solution
def maxSubsetSum(arr):
    def sum_factors(num):
        total = 0
        for i in range(1, int(num**0.5) + 1):
            if num % i == 0:
                total += i
                if i != num // i:
                    total += num // i
        return total
    
    return [sum_factors(x) for x in arr]
Question2
![image](https://github.com/user-attachments/assets/36ff02ae-e6c0-4dc1-bb8a-6f19255c602c)
![image](https://github.com/user-attachments/assets/459304b3-80fd-4f0b-85fe-03cf24e9047f)
![image](https://github.com/user-attachments/assets/76af1108-f514-41a3-a8a8-04cf54373726)
![image](https://github.com/user-attachments/assets/a9f09006-2cb6-43be-a2d4-5e9119153363)
#!/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'getMinOperations' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER_ARRAY arr
#  2. INTEGER_ARRAY change
#

def getMinOperations(arr, change):
    # Initialize steps and a flag to track progress
    steps = 0
    n = len(arr)
    nullified = [False] * n  # Tracks if an element is nullified

    # Keep processing until no more elements can be nullified
    while True:
        progress = False  # Tracks if we made progress in this iteration

        for i in range(n):
            if arr[i] > 0:
                arr[i] -= 1  # Decrement the element
                steps += 1  # Increment step count
            elif arr[i] == 0 and not nullified[i]:
                # Check dependency condition
                if change[i] > 0 and arr[change[i]] == 0:
                    nullified[i] = True  # Mark as nullified
                    steps += 1  # Increment for nullification
                    progress = True
                elif change[i] == 0:
                    nullified[i] = True  # Mark as nullified directly
                    steps += 1
                    progress = True

        # If no progress was made, stop the loop
        if not progress:
            break

    # Check if all elements are nullified
    if all(nullified):
        return steps
    else:
        return -1


if __name__ == '__main__':
    # Test input
    arr_test = [7, 2, 4, 7, 2, 3, 3, 5, 1, 1, 9, 3, 3, 10, 1, 2, 6, 8, 3, 5, 5, 1, 6, 9, 7, 6, 8, 7, 7]
    change_test = [7, 28, 17, 17, 29, 16, 24, 29, 12, 28, 29, 27, 14, 15, 12, 16, 8, 18, 16, 26, 16, 7, 11, 11, 20, 29, 19, 17, 15]

    # Run the function
    result = getMinOperations(arr_test, change_test)
    print("Result:", result)  # Expected output depends on the logic and conditions

OR
#!/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'getMinOperations' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER_ARRAY arr
#  2. INTEGER_ARRAY change
#

def getMinOperations(arr, change):
    # Initialize the total number of steps
    total_steps = 0

    # Debug: Print the input arrays for validation
    print("Array to nullify (arr):", arr)
    print("Change array (change):", change)

    # Calculate total steps for all elements in arr
    for num in arr:
        total_steps += (num + 1)  # Decrements + nullification

    # Debug: Print the total steps calculated
    print("Total steps:", total_steps)

    # Compare total steps to the allowed limit (change[0])
    if total_steps <= change[0]:
        return total_steps
    else:
        return -1


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    arr_count = int(input().strip())

    arr = []

    for _ in range(arr_count):
        arr_item = int(input().strip())
        arr.append(arr_item)

    change_count = int(input().strip())

    change = []

    for _ in range(change_count):
        change_item = int(input().strip())
        change.append(change_item)

    result = getMinOperations(arr, change)

    fptr.write(str(result) + '\n')

    fptr.close()


Question 3
![image](https://github.com/user-attachments/assets/82238071-49c2-48c0-9fc0-4c7bc794acad)
![image](https://github.com/user-attachments/assets/867a4211-bf70-4811-b6da-cec075920058)
solution
import java.util.HashSet;
import java.util.Set;

class Solution {
    public int solution(String S) {
        // Create a HashSet to store unique substrings
        Set<String> uniqueSubstrings = new HashSet<>();
        
        // Extract all three-letter substrings
        for (int i = 0; i <= S.length() - 3; i++) {
            String substring = S.substring(i, i + 3);
            uniqueSubstrings.add(substring);
        }
        
        // Return the size of the HashSet as the result
        return uniqueSubstrings.size();
    }
}

question 4
![image](https://github.com/user-attachments/assets/80ae092e-0829-45c3-b472-01f52fc7f811)
![image](https://github.com/user-attachments/assets/9180a81f-a869-48a2-b13b-260f3648af9f)
![image](https://github.com/user-attachments/assets/4b92a47c-fb84-476b-93fb-585fa77dce3b)
solution
class Solution {
    public String encrypt(String text) {
        StringBuilder encrypted = new StringBuilder();

        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            
            // Rotate the character by 4 positions
            char rotated = (char) ((ch - 'A' + 4) % 26 + 'A');
            
            encrypted.append(rotated);
        }

        return encrypted.toString();
    }
}

question 5
![image](https://github.com/user-attachments/assets/3bfba332-8203-41a5-95e8-d3bb0acf756d)
![image](https://github.com/user-attachments/assets/51a5bc38-a718-4f95-9e42-4b79299bd0c8)
![image](https://github.com/user-attachments/assets/d0f0a06d-4624-4b3e-b09e-5bc7edad05fb)
Level 1 Sol
from cloud_storage import CloudStorage
from typing import Optional

class CloudStorageImpl(CloudStorage):
    def __init__(self):
        self.storage = {}  # Initialize storage as an empty dictionary

    def add_file(self, name: str, size: int) -> bool:
        if name in self.storage:
            return False  # File already exists
        self.storage[name] = size  # Add file with the specified size
        return True  # File added successfully

    def get_file_size(self, name: str) -> Optional[int]:
        return self.storage.get(name)  # Return file size if exists, otherwise None

    def delete_file(self, name: str) -> Optional[int]:
        return self.storage.pop(name, None)  # Remove file and return its size or None
level 2 question
![image](https://github.com/user-attachments/assets/56b67e1a-f4ae-41f8-8432-82f4f7e94209)
![image](https://github.com/user-attachments/assets/042403df-f1d3-4080-95ba-b717a4cd7812)
![image](https://github.com/user-attachments/assets/df5f48bd-d1d3-42a2-b06c-06b54af5824d)
from cloud_storage import CloudStorage
from typing import Optional

class CloudStorageImpl(CloudStorage):
    def __init__(self):
        # Initialize the storage dictionary
        self.storage = {}

    def add_file(self, name: str, size: int) -> bool:
        if name in self.storage:
            return False  # File already exists
        self.storage[name] = size
        return True  # File added successfully

    def get_file_size(self, name: str) -> Optional[int]:
        return self.storage.get(name)  # Return file size if it exists, else None

    def delete_file(self, name: str) -> Optional[int]:
        return self.storage.pop(name, None)  # Remove file and return its size, else None

    def get_n_largest(self, prefix: str, n: int) -> list[str]:
        # Filter files with names starting with the prefix
        filtered_files = [(name, size) for name, size in self.storage.items() if name.startswith(prefix)]

        # Sort by size (descending), and lexicographical order for ties
        sorted_files = sorted(filtered_files, key=lambda x: (-x[1], x[0]))

        # Format the output as "<name>(<size>)"
        formatted_files = [f"{name}({size})" for name, size in sorted_files[:n]]

        return formatted_files

Level 3 question
![image](https://github.com/user-attachments/assets/8c46d753-7d64-4cae-99dd-5945f3a9f226)
![image](https://github.com/user-attachments/assets/63cbfb06-7a22-4385-8414-fd589071001d)
![image](https://github.com/user-attachments/assets/3ffcc99d-c0fe-4ab4-b973-6557b64626a6)

from cloud_storage import CloudStorage
from typing import Optional


class CloudStorageImpl(CloudStorage):

    def __init__(self):
        self.storage = {}
        self.users = {}

    def add_file(self, name: str, size: int) -> bool:
        if name in self.storage:
            return False
        self.storage[name] = {"size": size, "owner": "admin"}
        return True

    def get_file_size(self, name: str) -> Optional[int]:
        file_info = self.storage.get(name)
        return file_info["size"] if file_info else None

    def delete_file(self, name: str) -> Optional[int]:
        file_info = self.storage.pop(name, None)
        return file_info["size"] if file_info else None

    def get_n_largest(self, prefix: str, n: int) -> list[str]:
        filtered_files = [
            (name, details["size"]) for name, details in self.storage.items() if name.startswith(prefix)
        ]
        sorted_files = sorted(filtered_files, key=lambda x: (-x[1], x[0]))
        formatted_files = [f"{name}({size})" for name, size in sorted_files[:n]]
        return formatted_files

    def add_user(self, user_id: str, capacity: int) -> bool:
        if user_id in self.users:
            return False
        self.users[user_id] = {"capacity": capacity, "used": 0}
        return True

    def add_file_by(self, user_id: str, name: str, size: int) -> Optional[int]:
        if user_id not in self.users:
            return None
        user = self.users[user_id]
        if user["used"] + size > user["capacity"]:
            return None
        if name in self.storage:
            return None
        self.storage[name] = {"size": size, "owner": user_id}
        user["used"] += size
        return user["capacity"] - user["used"]

    def merge_user(self, user_id_1: str, user_id_2: str) -> Optional[int]:
        if user_id_1 not in self.users or user_id_2 not in self.users:
            return None
        if user_id_1 == user_id_2:
            return None
        user1 = self.users[user_id_1]
        user2 = self.users[user_id_2]
        total_used = user1["used"] + user2["used"]
        total_capacity = user1["capacity"] + user2["capacity"]
        if total_used > total_capacity:
            return None
        for name, file_info in list(self.storage.items()):
            if file_info["owner"] == user_id_2:
                file_info["owner"] = user_id_1
        user1["capacity"] = total_capacity
        user1["used"] = total_used
        del self.users[user_id_2]
        return user1["capacity"] - user1["used"]




Question 6
![image](https://github.com/user-attachments/assets/cbcfe4ca-ae7f-4a30-a2f2-205e889bd617)
![image](https://github.com/user-attachments/assets/e5b20cec-bcf3-490b-af9f-72d2eed03464)
![image](https://github.com/user-attachments/assets/e9c3a9b6-2414-4e70-8a27-db98b133f65a)
![image](https://github.com/user-attachments/assets/17c53256-a1db-428d-9738-9c8128f774e4)


solution
def solution_with_dynamic_updates(S):
    """
    Solve the problem using dynamic updates for occupied positions, explicitly handling sequential dependencies.
    
    Args:
    - S: String representing player moves.

    Returns:
    - successful_moves: Total number of successful moves.
    """
    occupied_positions = {i for i in range(len(S))}  # Start with all players occupying their positions
    successful_moves = 0

    for idx, direction in enumerate(S):
        # Determine the target position based on the direction
        if direction == '<':
            target = idx - 1
        elif direction == '>':
            target = idx + 1
        elif direction in ('^', 'v'):  # Vertical moves are always successful
            successful_moves += 1
            occupied_positions.discard(idx)  # Free current position
            continue
        else:
            continue  # Ignore invalid directions

        # Check if the target position is valid and unoccupied
        if target not in occupied_positions:
            successful_moves += 1  # Increment successful moves
            occupied_positions.discard(idx)  # Free current position
            occupied_positions.add(target)  # Occupy the new position

    return successful_moves

# Define test cases
test_cases = [
    ("><^v", 2),
    ("<<^<v>>", 6),
    ("><><", 0),
]

# Run and collect test results
results = []
for test_input, expected_output in test_cases:
    computed_output = solution_with_dynamic_updates(test_input)
    results.append((test_input, expected_output, computed_output, computed_output == expected_output))

import pandas as pd
import ace_tools as tools

# Display results
df = pd.DataFrame(results, columns=["Test Input", "Expected Output", "Computed Output", "Pass/Fail"])
tools.display_dataframe_to_user(name="Test Results Using Dynamic Updates", dataframe=df)
Question 7
![image](https://github.com/user-attachments/assets/f7ada48b-fb17-4419-8035-6a75465d008f)
![image](https://github.com/user-attachments/assets/cc7dec5b-68c5-494c-93f8-cd09b8de91ef)
![image](https://github.com/user-attachments/assets/0263133d-9f94-4299-b6fd-2c178f3fc1af)

Solution
def solution(A):
    r = 0
    i = 0
    while i < len(A):
        j = i
        while i < len(A) and A[i] == A[j]:
            i += 1
        # A[j] occurs (i - j) times.
        occurrences = i - j
        r += min(abs(A[j] - occurrences), occurrences)
    return r

# Test cases
test_cases = [
    ([1, 1, 3, 4, 4, 4], 3),  # Expected: 3
    ([1, 2, 2, 2, 5, 5, 5, 8], 4),  # Expected: 4
    ([1, 1, 1, 1, 3, 3, 4, 4, 4, 4, 4], 5),  # Expected: 5
    ([10, 10, 10], 3),  # Expected: 3
]

# Evaluate test cases
test_results = []
for test_input, expected in test_cases:
    result = solution(test_input)
    test_results.append((test_input, result, expected, result == expected))

test_results

Question 7
![image](https://github.com/user-attachments/assets/0397cc0d-d2ee-42cf-b64d-6e5f0b5e3305)
![image](https://github.com/user-attachments/assets/b80f8051-4b0f-4515-9788-464a6b2aeee6)
![image](https://github.com/user-attachments/assets/1b526368-6722-4203-a8fc-4c5f660a32b4)
![image](https://github.com/user-attachments/assets/e3d70207-cff2-49ef-ad5e-d4a97f0c9c7d)
from collections import Counter

def solution(A, B):
    """
    Calculate the number of corresponding fragments between strings A and B.
    
    Args:
    - A: string of length N
    - B: string of length N
    
    Returns:
    - int: number of corresponding fragments
    """
    n = len(A)
    count = 0

    for i in range(n):
        for j in range(i + 1, n + 1):  # Fragment lengths range from 1 to n
            fragment_a = A[i:j]
            fragment_b = B[i:j]
            # Check if fragments are anagrams by comparing character counts
            if Counter(fragment_a) == Counter(fragment_b):
                count += 1

    return count


# Test cases
test_cases = [
    ("dBacaAA", "caBdaaA", 5),
    ("zzzX", "zzzX", 10),
    ("abc", "ABC", 0),
    ("ZZXYOYZ", "OOYXZZZ", 2),
]

results = []
for A, B, expected in test_cases:
    result = solution(A, B)
    results.append((A, B, expected, result, result == expected))

# Display results in a DataFrame
import pandas as pd
df = pd.DataFrame(results, columns=["String A", "String B", "Expected Output", "Computed Output", "Pass/Fail"])
import ace_tools as tools; tools.display_dataframe_to_user(name="Corresponding Fragments Results", dataframe=df)

Question 8
![image](https://github.com/user-attachments/assets/6b5bae9e-1837-4d75-805a-6223b57ea1db)
![image](https://github.com/user-attachments/assets/8ae0b932-0730-4ca3-9693-a5b463f616f1)
![image](https://github.com/user-attachments/assets/a1b2ceec-3a9a-4df1-80be-ede72364a41f)
Solution
from itertools import permutations

def solution(A, B, C, D):
    """
    Count the number of valid times that can be formed using the given digits.
    
    Args:
    - A, B, C, D: Integers representing the four digits.
    
    Returns:
    - int: Number of valid times.
    """
    digits = [A, B, C, D]
    valid_times = 0

    # Generate all permutations of the four digits
    for perm in permutations(digits):
        hours = perm[0] * 10 + perm[1]  # First two digits as hours
        minutes = perm[2] * 10 + perm[3]  # Last two digits as minutes

        # Check if the time is valid (24-hour format)
        if 0 <= hours < 24 and 0 <= minutes < 60:
            valid_times += 1

    return valid_times


# Test cases
test_cases = [
    (1, 8, 3, 2, 6),  # Example 1: 6 valid times
    (2, 3, 3, 2, 3),  # Example 2: 3 valid times
    (6, 2, 4, 7, 0),  # Example 3: 0 valid times
]

results = []
for A, B, C, D, expected in test_cases:
    result = solution(A, B, C, D)
    results.append((A, B, C, D, expected, result, result == expected))

# Display results in a DataFrame
df = pd.DataFrame(results, columns=["A", "B", "C", "D", "Expected Output", "Computed Output", "Pass/Fail"])
tools.display_dataframe_to_user(name="Valid Time Results", dataframe=df)

def validate(A, B, C, D):
    """
    Validate if the given 4 digits form a valid 24-hour time.
    
    Args:
    - A, B, C, D: Integers representing the digits of a time.
    
    Returns:
    - bool: True if the digits form a valid time, False otherwise.
    """
    hours = A * 10 + B
    minutes = C * 10 + D
    return 0 <= hours < 24 and 0 <= minutes < 60


def permutations(arr):
    """
    Generate all permutations of an array without importing a library.
    
    Args:
    - arr: List of integers to permute.
    
    Returns:
    - List[List[int]]: All permutations of the input array.
    """
    if len(arr) <= 1:
        return [arr]
    perm_list = []
    for i in range(len(arr)):
        remaining = arr[:i] + arr[i+1:]
        for perm in permutations(remaining):
            perm_list.append([arr[i]] + perm)
    return perm_list


def unique(arr):
    """
    Filter unique permutations from a list of permutations.
    
    Args:
    - arr: List of permutations (list of lists).
    
    Returns:
    - List[List[int]]: Unique permutations.
    """
    unique_perms = []
    for d in arr:
        if d not in unique_perms:
            unique_perms.append(d)
    return unique_perms


def solution(A, B, C, D):
    """
    Count the number of valid times that can be formed using the given digits.
    
    Args:
    - A, B, C, D: Integers representing the four digits.
    
    Returns:
    - int: Number of valid times.
    """
    perms = unique(permutations([A, B, C, D]))
    return sum(1 for perm in perms if validate(*perm))


# Test cases
test_cases = [
    (2, 2, 5, 9),  # Expected: Valid combinations exist
    (0, 0, 0, 0),  # Expected: Only one valid time (00:00)
    (1, 2, 3, 4),  # Expected: Multiple valid times
    (5, 5, 5, 5),  # Expected: No valid times
    (1, 1, 1, 2),  # Expected: Few valid times
]

# Run test cases
results = []
for A, B, C, D in test_cases:
    results.append((A, B, C, D, solution(A, B, C, D)))

# Display results
df = pd.DataFrame(results, columns=["A", "B", "C", "D", "Valid Times"])
tools.display_dataframe_to_user(name="Valid Time Results for Given Test Cases", dataframe=df)






















