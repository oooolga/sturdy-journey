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
