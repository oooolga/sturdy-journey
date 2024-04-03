# First Missing Positive

## Description
Given an unsorted integer array `nums`. Return the smallest positive integer that is not present in `nums`. 

**Run time: O(n); Space: O(1)**

## Hint
One way to address this problem is by conceptualizing it as a seating arrangement challenge, where the objective is to position each number at its designated index—meaning the number 1 should occupy the first position in `nums`, and the number 5 should be in the fifth position of `nums`.

In this reorganized list, the solution is identified as the first element whose value does not match the value of its corresponding index.

## Solution
Let `n` represent the upper bound of the potential solution, initialized to the length of the array `nums`. Beginning with the first element, if it qualifies as a valid element—specifically, if `nums[0] > 0 and nums[0] <= n`, we swap `nums[0]` with `nums[nums[0] - 1]`. This swapping operation is designed to place the original `nums[0]` in its correct position, according to its value. If `nums[0]` does not qualify as a valid element, we swap it with the last potential valid element from the end of the array (in this case, it is the last element of `nums`).

If `nums[0]` fails to qualify as a valid element—meaning it does not satisfy the condition `nums[0] > 0 and nums[0] <= n`, then we swap it with what is considered the last potential valid element in the array, which, in this context, is the element at the end of nums (i.e., `nums[n-1]`). Then, we decrease the uppoer bound of the potential solution(`n`)'s value by 1.

`nums[0]`, is swapped either with the element at the position `nums[nums[0]-1]` or with the element at the last position, `nums[n-1]`. This swap results in a new element being placed at the first position (`nums[0]`). The process is then repeated with this new element now at the starting position. This iterative process continues until either the number `1` is positioned at `nums[0]`, or the upper bound `n` is reduced to `1`.

If the upper bound `n` has not reduced to `1`. We proceed to the array's second element, aiming to position the number `2` into the second slot by using the same swapping algorithm.

We continue to apply this process iteratively, ensuring each element is correctly positioned in the array according to its value.

```
class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        counter = 0
        last_counter, invalid_counter = 0, 0
        while counter < n-last_counter:
            curr_val = nums[counter]
            while curr_val != counter+1:
                if (curr_val > 0 and curr_val < n-last_counter) and curr_val != nums[curr_val-1]:
                    nums[counter], nums[curr_val-1] = nums[curr_val-1], nums[counter]
                else:
                    last_counter += 1
                    nums[counter], nums[-last_counter] = nums[-last_counter], nums[counter]
                    if nums[-last_counter] != n-last_counter+1:
                        invalid_counter = last_counter
                    if counter == n-last_counter:
                        break       
                curr_val = nums[counter]
            counter += 1
            if counter >= n-invalid_counter:
                return n-invalid_counter + 1  
        return n-invalid_counter + 1
```