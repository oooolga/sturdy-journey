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
