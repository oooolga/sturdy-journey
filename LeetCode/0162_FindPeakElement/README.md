# [Find Peak Element](https://leetcode.com/problems/find-peak-element/)

## Description
A peak element is an element that is strictly greater than its neighbors.

Given a 0-indexed integer array `nums`, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.

You may imagine that `nums[-1] = nums[n] = -∞`. In other words, an element is always considered to be strictly greater than a neighbor that is outside the array.

You must write an algorithm that runs in `O(log n)` time.

## Solution

I use a binary search approach to find the peak element. I split the list at the midpoint and compare the middle element with its neighbor. If the element at the midpoint is larger than both of its neighbours, it is a peak element. Else if the element at the midpoint is smaller than its right neighbor, then there must be a peak in the right half; otherwise, a peak exists in the left half. This works because a rising sequence guarantees a peak, while a falling sequence implies that a peak was passed along the way.

```
class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        

        def FPE_recursion(nums, left_i, right_i):

            if left_i == right_i:
                return left_i
            elif nums[left_i+1] < nums[left_i]:
                # we found a peak element at the left end of the list
                return left_i
            elif nums[right_i-1] < nums[right_i]:
                return right_i
            else:
                # the list length must be >=3
                half_i = (right_i+left_i) // 2
                if nums[half_i] > nums[half_i-1] and nums[half_i] > nums[half_i+1]:
                    return half_i
                elif nums[half_i] < nums[half_i-1]:
                    return FPE_recursion(nums, left_i=left_i, right_i=half_i-1)
                else:
                    # num[half_i] < nums[half_i+1]
                    return FPE_recursion(nums, left_i=half_i+1, right_i=right_i)
        
        return FPE_recursion(nums, 0, len(nums)-1)
```