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