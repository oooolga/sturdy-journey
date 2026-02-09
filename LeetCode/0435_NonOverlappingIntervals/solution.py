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