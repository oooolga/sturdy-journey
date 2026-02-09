# [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/)
Given an array of intervals intervals where `intervals[i] = [starti, endi]`, return the *minimum number* of *intervals* you need to remove to make the rest of the intervals non-overlapping.

Note that intervals which only touch at a point are **non-overlapping**. For example, `[1, 2]` and `[2, 3]` are non-overlapping.

## Solution
I solve this using a Greedy approach by first sorting the intervals based on their start times. I maintain a `current_end` buffer to track the end of the last accepted interval. As I iterate through the list, if a new interval's start time is greater than or equal to the `current_end`, I update the buffer to the new end time. If an overlap occurs (start time is less than `current_end`), I increment the removal count and greedily retain the interval with the earlier end time to maximize the remaining space for future intervals.

## Code
```Py
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort() # nlog(n); sorting by first element (start_i)
        end_time = intervals[0][1]
        ret = 0
        
        for i in range(1, len(intervals)):
            interval = intervals[i]
            if interval[0] < end_time:
                # if overlap!
                ret += 1
                end_time = min(interval[1], end_time)
            else:
                # no overlap; update end_time
                end_time = interval[1]
        return ret
```