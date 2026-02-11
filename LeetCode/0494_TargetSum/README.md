# [Target Sum](https://leetcode.com/problems/target-sum/)

You are given an integer array `nums` and an integer `target`.

You want to build an expression out of `nums` by adding one of the symbols `'+'` and `'-'` before each integer in nums and then concatenate all the integers.

- For example, if `nums = [2, 1]`, you can add a `'+'` before `2` and a `'-'` before `1` and concatenate them to build the expression `"+2-1"`.

Return the number of different expressions that you can build, which evaluates to `target`.

## Solution
First, we iterate through `nums`. At each step, we maintain a dictionary `prev_possible_sums` where the *keys* represent all achievable intermediate sums using the numbers processed so far, and the *values* store how many ways each sum can be formed. After processing all numbers, we return `prev_possible_sums[target]`.

## Code
```Py
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:
        from collections import defaultdict

        prev_possible_sums = {0:1}

        for num in nums:
            possible_sums = {}

            for key in prev_possible_sums.keys():
                target_value1 = key+num
                target_value2 = key-num
                
                if target_value1 == target_value2: # num == 0
                    possible_sums[key] = prev_possible_sums[key] * 2
                else:
                    # Let's focus on target_value1 first
                    if not target_value1+num in prev_possible_sums:
                        # we can only get target2 with current key
                        possible_sums[target_value1] = prev_possible_sums[key]
                    else:
                        # we can get target2 with current key or with a key==target_value1
                        possible_sums[target_value1] = prev_possible_sums[key] + prev_possible_sums[target_value1+num]
                    
                    if not target_value2-num in prev_possible_sums:
                        # we can only get target2 with current key
                        possible_sums[target_value2] = prev_possible_sums[key]
                    else:
                        # we can get target2 with current key or with a key==target_value1
                        possible_sums[target_value2] = prev_possible_sums[key] + prev_possible_sums[target_value2-num]
            prev_possible_sums = possible_sums

        return prev_possible_sums[target] if target in prev_possible_sums.keys() \
               else 0
```