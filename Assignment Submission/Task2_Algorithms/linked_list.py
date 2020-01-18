class linked_list:
    def __init__(self,x):
        self.val = x
        self.next = None

    def next_list(self,next_list):
        self.next = next_list


#
class Solution:
    def mergeKlists1(self,lists): # Complexity O(N), Only suitable for sorted linked list
        '''
        Merge k sorted linked list

        Arguments:
        lists -- example: [x,y,z]
        x >>> the first element of the linked_list1 >>> x.next is the 2nd element of linked_list1
        y >>> the first element of the linked_list2 >>> y.next is the 2nd element of linked_list2
        z >>> the first element of the linked_list3 >>> z.next is the 2nd element of linked_list3

        example:
        []
        Return:
            L : The first element of sorted merged linked list  >>> L.next is the 2nd element of sorted merged linked list
            Traverse the whole linked list by using while loop:
                while L is not None:
                    print(L.val)
                    L = L.next
        '''
        x = []                                                          # Initialize a list to store the new sorted linked list
        while lists:                                                    # Compare one element in each linked list separately
            min_val = min([lists[i].val for i in range(len(lists))])    # Get the minimum value of them
            for i in range(len(lists)):                                 # Traverse each linked list to find who has the minimum
                if lists[i].val == min_val:
                    x.append(lists[i])                                  # Append the node of that element to the new list
                    lists[i] = lists[i].next                            # Move one step forward in that linked list
                                                                        # and go back to compare with elements in other linked list
                    if lists[i] == None:                                # until one linked list comes to its end
                        lists.remove(lists[i])                          # Remove that linked list and continue the while loop
                    break                                               # Until no element is left in the original list
        x_length = len(x)-1                                             # Now we have all the sorted node in list x
        for i in range(x_length):                                       # we need to linked them together to get a new soted linked list
            x[i].next = x[i+1]
        x[x_length].next = None
        return x[0]                                                     # return the first element of the sorted merged linked list










    def mergeKlists2(self,lists):    # Complexity O(N*logN) But it is also suitable for mering K unsorted linked list
        '''
        Merge k sorted linked list

        Arguments:
        lists -- example: [x,y,z]
        x >>> the first element of the linked_list1 >>> x.next is the 2nd element of linked_list1
        y >>> the first element of the linked_list2 >>> y.next is the 2nd element of linked_list2
        z >>> the first element of the linked_list3 >>> z.next is the 2nd element of linked_list3

        example:
        []
        Return:
            L : The first element of sorted merged linked list  >>> L.next is the 2nd element of sorted merged linked list
            Traverse the whole linked list by using while loop:
                while L is not None:
                    print(L.val)
                    L = L.next
                '''
        v_map = {}                                              # Initialize a dictionary to store the node

        for element in lists:                                   # For every node in each linked list, create a key (the value of the node) and value (the node itself_
            while element:
                try:                                            # Use a list for each key to store node
                    v_map[element.val].append(element)
                except KeyError:                                # when enconuter the key the first time, create a list for futher appending other node with the same key
                    v_map[element.val] = [element]
                element = element.next

        key_sorted = sorted(v_map.keys())                       #Sorted all the key in the dic
        first_key = key_sorted[0]
        head = end = v_map[first_key][0]                        # identify the head

        for key in key_sorted:                                  # Create a new linked list following the order of key
            for node in v_map[key]:
                end.next = node                                 # Not need to worry the node point to itself at the first time. It will overwrite itself along with the loop.
                end = node
        end.next=  None
        return head

if __name__== '__main__':
    # To test the algorithm just type python3 linked_list.py in the terminal
    x1 = linked_list(1)
    x2 = linked_list(2)
    x3 = linked_list(5)

    x4 = linked_list(1)
    x5 = linked_list(4)
    x6 = linked_list(9)

    x7 = linked_list(3)
    x8 = linked_list(7)

    x1.next_list(x2)
    x2.next_list(x3)

    x4.next_list(x5)
    x5.next_list(x6)

    x7.next_list(x8)

    lists = [x1,x4,x7]
    s= Solution()
    result = s.mergeKlists1((lists) #you can also try s.mergeKlist2


    while result is not None:
        print(result.val)
        result = result.next