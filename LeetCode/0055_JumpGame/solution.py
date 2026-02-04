class Solution(object):
    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        lastReachableIdx = 0
        for i, num in enumerate(nums):
            if i > lastReachableIdx:
                return False
            
            lastReachableIdx = max(lastReachableIdx, i + num)
            if lastReachableIdx >= len(nums) - 1:
                return True