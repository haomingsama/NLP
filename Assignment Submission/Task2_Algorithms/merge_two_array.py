# def merge_naive(nums1, nums2):
#     p1= 0; p2= 0
#     new_list = []
#     m = len(nums1);n = len(nums2)
#
#     while True:
#         if nums1[p1] >= nums2[p2]:
#             new_list.append(nums2[p2])
#             p2 +=1
#         else:
#             new_list.append(nums1[p1])
#             p1 += 1
#
#         if p1 >m-1:
#             new_list = new_list+nums2[p2:]
#             break
#         if p2 >n-1:
#             new_list = new_list+ nums1[p1:]
#             break
#     return new_list

def merge_true(nums1, nums2,m,n):
    """
    Merge tow sorted array into one sorted array

    Arguments:
    nums1 -- example: [1,2,3,0,0,0]    m = 3
    nums2 -- example: [2,5,6]       n =3

    Return:
    nums1(after merge) -- [1,2,2,3,5,6]
    """
    p1 = 0 ; p2 = 0                                     # initialize two pointer
    while True:
        if nums1[p1] <= nums2[p2]:                      # move forward the pointer if nums1's element is smaller
            p1 += 1
        else:
            for i in range(m+n - 2, p1 - 1,-1):         # let the iterator start from the last 2nd position
                nums1[i + 1] = nums1[i]                 # move every element forward
            nums1[p1] = nums2[p2]                       # for clearing one position for the element in nums2
            p2 += 1
        if p1 > m-1:                                    # until p1 pointer surpass the end of nums1 (before the 1st 0)
            for i in range(m, m + n):                   # move every remaining element of nums2(start from p2) into the remaining position of nums1
                nums1[i] = nums2[p2]
                p2 += 1
            break

        if p2 >n-1:                                     # if p2 surpass the end of nums2
            break                                       # it means every element in nums2 has been moved into nums1
    return nums1

if __name__ == '__main__':
    #To test the algorithm just type python3 merge_two_array.py in the terminal
    import numpy as np
    # Create nums1 that contains 0 in the end
    nums1 = sorted(list(np.random.randint(10000, size =20)))
    nums0 = list(np.zeros(5))
    nums1= nums1+nums0
    # Create nums2
    nums2 = sorted(list(np.random.randint(1000, size=5)))
    print('Finish input initialization')
    print('nums1: ',nums1)
    print('nums2: ',nums2)
    print('Start Merging two array')
    print('Finish Merging : ',merge_true(nums1,nums2,20,5))

