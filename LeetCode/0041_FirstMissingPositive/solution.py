class Solution:
    def firstMissingPositive(self, nums):
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

if __name__ == "__main__":
    case1 = [1,2,0]
    case2 = [3,4,-1,1]
    case3 = [7,8,9,11,12]

    sol = Solution()
    print(sol.firstMissingPositive(case1))
    print(sol.firstMissingPositive(case2))
    print(sol.firstMissingPositive(case3))
