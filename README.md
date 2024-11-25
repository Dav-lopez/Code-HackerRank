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

Question 9
![image](https://github.com/user-attachments/assets/08a0cb0d-a50b-403b-95a6-897ac793e272)
![image](https://github.com/user-attachments/assets/a3a3737d-857d-4646-a48f-1f26ba39ebe4)
![image](https://github.com/user-attachments/assets/8f3d53cb-3553-44a3-9d19-284733de307f)
Solt
#!/bin/python3

import math
import os
import random
import re
import sys
from collections import Counter

#
# Complete the 'maximizeProcessingPower' function below.
#
# The function is expected to return a LONG_INTEGER.
# The function accepts INTEGER_ARRAY processingPower as parameter.
#

def maximizeProcessingPower(processingPower):
    # Count frequencies of processing powers
    freq = Counter(processingPower)
    uniqueValues = sorted(freq.keys())
    
    # DP variables
    prev2, prev1 = 0, 0
    
    for i in range(len(uniqueValues)):
        currentValue = uniqueValues[i] * freq[uniqueValues[i]]
        if i > 0 and uniqueValues[i] == uniqueValues[i - 1] + 1:  # Corrected variable name
            # Conflict: Either include current or skip it
            temp = max(prev1, prev2 + currentValue)
        else:
            # No conflict: Add current value
            temp = prev1 + currentValue
        prev2, prev1 = prev1, temp
    
    return prev1

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    processingPower_count = int(input().strip())

    processingPower = []

    for _ in range(processingPower_count):
        processingPower_item = int(input().strip())
        processingPower.append(processingPower_item)

    result = maximizeProcessingPower(processingPower)

    fptr.write(str(result) + '\n')

    fptr.close()

Question 10
![image](https://github.com/user-attachments/assets/6ed0527f-72ac-4be0-af51-201606fc2b49)
![image](https://github.com/user-attachments/assets/778dfe41-cf04-48b7-b279-e1ad9b2d1b69)
![image](https://github.com/user-attachments/assets/fdc55265-58be-4765-abee-e56a66f2b5c7)
![image](https://github.com/user-attachments/assets/e9540c9d-5485-429b-8d39-c2b355222bf8)
#!/bin/python3

import math
import os
import random
import re
import sys


#
# Complete the 'createMaximumCollaborations' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER_ARRAY creatorsEngagementPower
#  2. INTEGER minCreatorsRequired
#  3. LONG_INTEGER minTotalEngagementPowerRequired
#

def createMaximumCollaborations(creatorsEngagementPower, minCreatorsRequired, minTotalEngagementPowerRequired):
    n = len(creatorsEngagementPower)
    max_collab = 0
    i = 0

    while i <= n - minCreatorsRequired:
        team_power = 0
        team_size = 0

        for j in range(i, n):
            team_power += creatorsEngagementPower[j]
            team_size += 1

            if team_size >= minCreatorsRequired and team_power >= minTotalEngagementPowerRequired:
                max_collab += 1
                i = j + 1  # Skip past the current team
                break
        else:
            i += 1  # Increment `i` when no valid team is formed

    return max_collab


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    creatorsEngagementPower_count = int(input().strip())

    creatorsEngagementPower = []

    for _ in range(creatorsEngagementPower_count):
        creatorsEngagementPower_item = int(input().strip())
        creatorsEngagementPower.append(creatorsEngagementPower_item)

    minCreatorsRequired = int(input().strip())

    minTotalEngagementPowerRequired = int(input().strip())

    result = createMaximumCollaborations(creatorsEngagementPower, minCreatorsRequired, minTotalEngagementPowerRequired)

    fptr.write(str(result) + '\n')

    fptr.close()
Quest 11
![image](https://github.com/user-attachments/assets/5880608d-13d1-4316-be0d-83a889f99418)
![image](https://github.com/user-attachments/assets/c5f82843-6b86-4fdf-be8d-9b9f0a0478f7)
![image](https://github.com/user-attachments/assets/03528060-46f5-4b2e-bf53-e376ffe14191)
![image](https://github.com/user-attachments/assets/08cf915f-6a25-4557-8d39-7c82dd0a2731)
![image](https://github.com/user-attachments/assets/bd2d721e-310e-4085-86c3-6e8228f075a2)

Solution
def getTime(conversation, infectedCategory):
    # Efficient simulation of the infection propagation
    time = 0
    conversation = list(conversation)

    while True:
        found_infected = False
        to_remove = set()  # Set to store indices to remove

        # Identify positions to remove
        for i in range(1, len(conversation)):
            if conversation[i] == infectedCategory:
                to_remove.add(i - 1)  # Add the left neighbor to removal set
                found_infected = True

        # Remove marked indices
        conversation = [conversation[i] for i in range(len(conversation)) if i not in to_remove]

        if not found_infected:  # Stop if no more infections
            break

        time += 1  # Increment time for each propagation step

    return time


# Test cases
test_case_1 = ("pguxvg", 'v')  # Expected: 4
test_case_2 = ("abcdaed", 'd')  # Expected: 3
test_case_3 = ("bbbbb", 'b')    # Expected: 1
test_case_4 = ("abcd", 'e')     # Expected: 0
test_case_5 = ("aaaaa", 'a')    # Expected: 1

# Run tests
results = {
    "Test Case 1": getTime(*test_case_1),
    "Test Case 2": getTime(*test_case_2),
    "Test Case 3": getTime(*test_case_3),
    "Test Case 4": getTime(*test_case_4),
    "Test Case 5": getTime(*test_case_5),
}

results

Quest 12
![image](https://github.com/user-attachments/assets/3b3ee954-6cdd-4b86-8eaa-f1356703b9d8)
![image](https://github.com/user-attachments/assets/de501593-c0e2-48dc-8e92-98e5babecd79)

sol
from math import comb

def diverseDeputation(m, w):
    # Base cases: not enough people or diversity
    if m < 1 or w < 1 or m + w < 3:
        return 0

    # Case 1: 2 men and 1 woman
    men_case = comb(m, 2) if m >= 2 else 0
    women_case = comb(w, 1) if w >= 1 else 0

    # Case 2: 1 man and 2 women
    man_case = comb(m, 1) if m >= 1 else 0
    women_case_2 = comb(w, 2) if w >= 2 else 0

    # Total combinations
    total_combinations = men_case * women_case + man_case * women_case_2
    return total_combinations


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    m = int(input().strip())
    w = int(input().strip())

    result = diverseDeputation(m, w)

    fptr.write(str(result) + '\n')
    fptr.close()


Question 13
![image](https://github.com/user-attachments/assets/d9a67a17-9746-44a5-b208-a9bf3cdd79c4)
![image](https://github.com/user-attachments/assets/9da5f187-7592-4cbd-80a1-94db15b31c12)
![image](https://github.com/user-attachments/assets/a11a91dd-005b-4de9-a55d-ea9ce63310f6)
Solution
from itertools import compress

def getTime(conversation, infectedCategory):
    time = 0
    infected_positions = set()

    # Track initial infected positions
    for i in range(1, len(conversation)):
        if conversation[i] == infectedCategory:
            infected_positions.add(i - 1)

    # Process until no more infections occur
    while infected_positions:
        keep = [True] * len(conversation)
        new_infected_positions = set()
        found_infected = False

        # Update positions of new infections
        for pos in infected_positions:
            if pos > 0 and conversation[pos - 1] != infectedCategory:
                new_infected_positions.add(pos - 1)
                keep[pos - 1] = False
                found_infected = True

        # Update the infected positions for the next iteration
        infected_positions = new_infected_positions
        if not found_infected:
            break
        time += 1

    return time












