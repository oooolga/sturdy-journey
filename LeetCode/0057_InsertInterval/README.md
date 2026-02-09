# [Insert Interval](https://leetcode.com/problems/insert-interval/description/)

You are given an array of non-overlapping intervals intervals where `intervals[i] = [starti, endi]` represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval `newInterval = [start, end]` that represents the start and end of another interval.

Insert `newInterval` into `intervals` such that `intervals` is still sorted in ascending order by `starti` and `intervals` still does not have any overlapping intervals (merge overlapping intervals if necessary).

Return `intervals` after the insertion.

Note that you don't need to modify `intervals` in-place. You can make a new array and return it.

## Solution
The algorithm processes the sorted intervals in three distinct phases. First, it **appends all intervals that end before the `newInterval` starts**. Second, it **merges all overlapping intervals** by iteratively updating the `newInterval`'s start and end to the minimum and maximum of the overlapping bounds. Finally, once an interval is found that starts after the `newInterval` ends, the merged `newInterval` is added to the result, followed by the **remaining non-overlapping intervals**.

## Code
```Python
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        merged = []
        pointer = 0

        while pointer < len(intervals):
            curr_interval = intervals[pointer]
            if curr_interval[1] < newInterval[0]:
                # curr_interval ends before newInterval starts
                merged.append(curr_interval)
                pointer += 1
            else:
                break
        
        while pointer < len(intervals):
            # curr_interval ends after newInterval starts; merge happens
            curr_interval = intervals[pointer]
            newInterval[0] = min(curr_interval[0], newInterval[0])

            if newInterval[1] <= curr_interval[1]:
                # NewInterval ends before curr_interval ends
                if newInterval[1] < curr_interval[0]:
                    # NewInterval ends before curr_interval starts
                    break
                newInterval[1] = curr_interval[1]
                pointer += 1
                break
            pointer += 1
        merged.append(newInterval)

        return merged + intervals[pointer:]
```