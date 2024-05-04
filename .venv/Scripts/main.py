from typing import List
from typing import Optional,Tuple
from heapq import heappush, heappop
import builtins
from collections import deque,defaultdict


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class ListNode:
    def __init__(self,val=0,next=None):
        self.val=val
        self.next=next


class Solution:
    #399. Evaluate Division
    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        """
        a/b = 2; b/a = 1/2
        b/c = 3; c/b = 1/3
        a/c = a/b * b/c = 2 * 3 = 6
        """
        # Construct graph
        # Reciprocal value enables bidirectional access
        n = len(equations)
        graph = defaultdict(dict)
        for idx, (src, dest) in enumerate(equations):
            graph[src][dest] = values[idx]
            graph[dest][src] = 1 / values[idx]

        # DFS to process the result of queries
        def dfs(src, dest, res):
            if src not in graph or dest not in graph or src in visited:
                return -1
            if src == dest:
                return res
            visited.add(src)
            for nei, val in graph[src].items():
                temp = dfs(nei, dest, res * val)
                if temp != -1:
                    return temp
            return -1

        # Traverse over the queries and store the processed queries in result list
        result = []
        for src, dest in queries:
            visited = set()
            val = dfs(src, dest, 1)
            result.append(val)

        return result

    # Reoder Routes to Make all Paths Lead to the City Zero
    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        # Firstly build graph
        graph=defaultdict(list[Tuple[int,bool]])
        for city_a, city_b in connections:
            graph[city_a].append((city_b,True))
            graph[city_b].append((city_a,False))
        # Perform BFS starting from node 0
        queue=deque([(int(0),int(0))])
        visited=[False]*n
        count=0
        while queue:
            city,n=queue.popleft()
            if visited[city]:
                continue
            visited[city]=True

            for next, is_forward in graph[city]:
                if visited[next]:
                    continue
                if is_forward:
                    count+=1
                queue.append((next,city))
        return count






    #bfc



            


    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        self.result=0
        self.helpZigZag(root,0,False)
        return self.result



    def helpZigZag(self,root,count,isLeft):
        if root is None:
            return
        if count>self.result:
            self.result=count
        if isLeft:
            self.helpZigZag(root.right,count+1,False)
            self.helpZigZag(root.right, 0,True)
        else:
            self.helpZigZag(root.left,count+1,True)
            self.helpZigZag(root.left, 0,False)



    def pathSum(self, root, target):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """
        # define global return var
        self.numOfPaths = 0
        # 1st layer DFS to go through each node
        self.dfs(root, target)
        # return result
        return self.numOfPaths

    # define: traverse through the tree, at each treenode, call another DFS to test if a path sum include the answer
    def dfs(self, node, target):
        # exit condition
        if node is None:
            return
            # dfs break down
        self.test(node, target)  # you can move the line to any order, here is pre-order
        self.dfs(node.left, target)
        self.dfs(node.right, target)

    # define: for a given node, DFS to find any path that sum == target, if find self.numOfPaths += 1
    def test(self, node, target):
        # exit condition
        if node is None:
            return
        if node.val == target:
            self.numOfPaths += 1

        # test break down
        self.test(node.left, target - node.val)
        self.test(node.right, target - node.val)


    #Bianary Tree BFS
    def goodNodes(self, root: TreeNode) -> int:
        
        def branchIsGood(root):
            if root == None:
                return 0
            elif root.left != None:
                if root.val <= root.left.val:
                    return 1+branchIsGood(root.left)
            elif root.right != None:
                if root.val <= root.right.val:
                    return 1+branchIsGood(root.right)
            else:
                return 1            

        
        return branchIsGood(root)

    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        def find_leaves(tree):
            if tree == None:
                return []
            elif tree.left == tree.right == None:
                return [tree.val]
            else:
                return find_leaves(tree.left) + find_leaves(tree.right)

        return find_leaves(root1) == find_leaves(root2)


    #LinkedList
    #328
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head == None:
            return None
        if head.next == None:
            return head
        headOdd = head
        headEven = head.next
        endOdd = head
        endEven = head.next
        count = 1
        cur = None
        if head.next != None:
            cur = head.next.next
        while cur != None:
            if count % 2 == 0:
                endEven.next = cur
                endEven = cur
            else:
                endOdd.next = cur## после 0 записываем 2
                endOdd = cur ## начало начинаем с 2-ки
            cur = cur.next
            count += 1
        endEven.next = None
        endOdd.next = headEven
        return headOdd



    #Stack
    def decodeString(self, s: str) -> str:
        stack = []
        for char in s:
            if char == "]":
                val = stack.pop()
                item = ""
                while stack and not str(val).isnumeric():
                    item = val + item if val != "[" else item
                    val = stack.pop()

                number = val
                while stack and str(stack[-1]).isnumeric():
                    number = stack.pop() + number
                print(number, item)
                [stack.append(item) for _ in range(int(number))]
            else:
                stack.append(char)

        return "".join(stack)

    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        ans=[]
        for one in asteroids:
            if not ans or one>0:
                ans.append(one)
            else:
                while True:
                    peek = ans[-1]
                    if peek < 0:
                        ans.append(one)
                        break
                    elif one ==-peek:
                        ans.pop()
                        break
                    elif abs(one)>peek:
                        ans.pop()
                        if len(ans)==0:
                            ans.append(one)
                            break
                        continue
                    else:
                        break
        return ans


    #2390
    def removeStars(self, s: str) -> str:
        ans=[]
        for i in s:
            if i=='*':
                ans.pop()
            else:
                ans+=[i]
        return "".join(ans)

    # HashMap
    # очень интересная задача с точки зрения питона!!!
    def equalPairs(self, grid: List[List[int]]) -> int:
        count = {}# создаем словарь
        result = 0
        for i in range(len(grid)):
            if tuple(grid[i]) not in count: # преобразуем строку в tuple
                count[tuple(grid[i])] = 1 # вставляем tuple в словарь если его нет
            elif tuple(grid[i]) in count:
                count[tuple(grid[i])] += 1 # если tuple в словарь то увеличиваем счетчик
        for n in range(len(grid)):
            col = [i[n] for i in grid] # выдергиваем столбец из массива
            if col in grid:# проверяем есть ли строка равная данному столбцу в массиве
                result += count[tuple(col)] # если есть прибавляем к результату количество строк данной строки
        return result



    
    def closeStrings(self, word1: str, word2: str) -> bool:
        freq1 = [0] * 26
        freq2 = [0] * 26
        for ch in word1:
            freq1[ord(ch) - ord('a')] += 1
        for ch in word2:
            freq2[ord(ch) - ord('a')] += 1
        for i in range(26):
            if (freq1[i] == 0 and freq2[i] != 0) or (freq1[i] != 0 and freq2[i] == 0):
                return False
        freq1.sort()
        freq2.sort()

        for i in range(26):
            if freq1[i] != freq2[i]:
                return False
        return True




    #2215
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        set1 = set(nums1)
        set2 = set(nums2)
        difference1 = list(set1 - set2)
        difference2 = list(set2 - set1)
        answer = [difference1, difference2]
        return answer

    def findDifferenceMy(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        common_integers = set(nums1).intersection(set(nums2))
        res0=set()
        res1=set()
        for i in range(len(nums1)):
            if not common_integers.__contains__(nums1[i]):
                res0.add(nums1[i])
        for i in range(len(nums2)):
            if not common_integers.__contains__(nums2[i]):
                res1.add(nums2[i])
        return [list(res0),list(res1)]
    
    #prefix Sum
    def pivotIndex(self, nums: List[int]) -> int:
        s1,s2=sum(nums),0
        left=0
        n=len(nums)
        while left<n-1:
            s2+=nums[left]
            if s2==(s1-nums[left+1])/2:
                return left+1
            left+=1
        return -1


    #Sliding window


    #1493
    def longestSubarray(self, nums: List[int]) -> int:
        l, zeros=0,0
        for r,n in enumerate(nums):
            zeros+=n==0
            if zeros>1:
                zeros-=nums[l]==0
                l+=1
        return r-l
# 1004. Max Consecutive Ones III
# Given a binary array nums and an integer k, return the maximum number of consecutive 1's in the array if you
# can flip at most k 0's.
    def longestOnes(self, nums: List[int], k: int) -> int:
        zeros, l = 0, 0
        for r, n in enumerate(nums):
            zeros += n == 0
            if zeros > k:
                zeros -= nums[l] == 0
                l += 1
        return r - l + 1

#1456 Maximum Number of Vowels in a Subsing of Given length
    def maxVowels(self, s: str, k: int) -> int:
        vowel = {"a", "e", "i", "o"}
        max=0;
        count=0
        for i in range(k):
            if vowel.__contains__(s[i]):
                count+=1
        m=count
        start=0;
        end=k
        while end<len(s):
            if(vowel.__contains__(s[start])):
                count-=1
            if(vowel.__contains__(s[end])):
                count+=1;
            start+=1
            end+=1
            m=max(m,count)
        return





    ##Two pointers solutions
    ##1679
    def maxOperations(self, nums: List[int], k: int) -> int:
        nums.sort()
        left=0
        right=len(nums)-1
        count=0
        while left<right:
            temp=nums[left]+nums[right]
            if temp==k:
                left+=1
                right-=1
                count+=1
            elif temp<k:
                left+=1
            else:
                right-=1
        return count

    def maxArea(self, height: List[int]) -> int:
        max_s = 0
        left = 0
        right = len(height) - 1
        while left < right:
            temp = (right - left) * min(height[left], height[right])
            if temp > max_s:
                max_s = temp
            elif height[left] < height[right]:
                left += 1
            else:
                right -= 1
        return max_s

    def isSubsequence(self, s: str, t: str) -> bool:
        first=0
        second=0
        while first<len(s) and second<len(t):
            if s[first]==t[second]:
                first=first+1
                second=second+1
                continue
            second=second+1
        return first==len(s)

    def moveZeroes(self, nums: List[int]) -> None:
        res=[]
        count=0
        for i in range(0,len(nums)):
            if nums[i]!=0:
                res.append(nums[i])
            else:
                count=count+1;
        for i in range(0,count):
            res.append(0)
        return res

    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        m=max(candies)
        res=[]
        for i in range(0,len(candies)):
            res.append(candies[i]+extraCandies>m)
        return res

    def minimizeStringValue(self, s: str) -> str:
        heap = []
        counts = {}

        for ch in 'abcdefghijklmnopqrstuvwxyz': counts[ch] = 0

        for ch in s:
            if ch != '?': counts[ch] += 1

        for ch in counts: heappush(heap, (counts[ch], ch))

        possible_ans = []

        for _ in range(s.count('?')):
            n, c = heappop(heap)
            possible_ans.append(c)
            heappush(heap, (n + 1, c))

        sort=sorted(possible_ans)

        possible_ans = deque(sorted(possible_ans))
        ans = list(s)

        for i in range(len(ans)):
            if ans[i] == '?': ans[i] = possible_ans.popleft()

        return "".join(ans)


    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        happiness.sort(reverse=True)
        m=min(len(happiness),k)
        res=happiness[0]
        for i in range(1,m):
            d=happiness[i]-i
            if  d<0:
                break
            res+=d
        return res


    def minimumBoxes(self, apple: List[int], capacity: List[int]) -> int:
        capacity.sort(reverse=True)
        sum0=sum(apple)
        count=1
        for i in range(0,len(capacity)):
            sum0=sum0-capacity[i]
            if sum0<=0:
                break
            count+=1
        return count




    def resultGrid(self, img: List[List[int]], threshold: int) -> List[List[int]]:
        n=len(img)
        m=len(img)
        x = 501  # Number of 2D arrays
        y = 501  # Number of rows in each 2D array
        z = 2 # Number of columns in each 2D array
        # Create a 3D array (list of lists of lists)
        reg = [[[0 for _ in range(z)] for _ in range(y)] for _ in range(x)]
        for i in range(0,n-2):
            for j in range(0,m-2):
                sum=0
                is_region=True
                for k in range(i,i+3):
                    for l in range(j,j+3):
                        sum =sum + img[k][l]
                        is_region = is_region and (k == i or abs(img[k][l] - img[k - 1][l]) <= threshold)
                        is_region = is_region and (l == j or abs(img[k][l] - img[k][l - 1]) <= threshold)
                if is_region:
                    for k in range(i,i+3):
                        for l in range(j,j+3):
                            reg[k][l][0]+=sum/9
                            reg[k][l][1]+=1
        for i in range(0,n):
            for j in range(0,m):
                if reg[i][j][1]:
                    img[i][j]=(int)(reg[i][j][0]/reg[i][j][1])
        return  reg









if __name__ == '__main__':

    sol=Solution()

    sol399=sol.calcEquation([["a","b"],["b","c"]],[2.0,3.0],[["a","c"],["b","a"],["a","e"],["a","a"],["x","x"]])

    sol1466_0=sol.minReorder(6,[[0,1],[1,3],[2,3],[4,0],[4,5]])

    trr = TreeNode(5, None, None)
    trl = TreeNode(1, None, None)
    rll = TreeNode(3, None, None)
    tll = TreeNode(3, None, None)
    tl = TreeNode(1, trl, None)
    tr = TreeNode(4, trr, trl)
    t = TreeNode(3, tl, tr)
    sol1372 = sol.longestZigZag(t);
    sol437=sol.pathSum(t,7);
    sol1448=sol.goodNodes(t);

    # LinkedList 328
    listL = ListNode(5, None)
    for i in range(4, -1, -1):
        cur = ListNode(i, listL)
        listL = cur
    sol328=sol.oddEvenList(listL)

    sol394=sol.decodeString("3[a]2[bc]")


    sol735_132 = sol.asteroidCollision([1,-2, -2])
    sol735_1 = sol.asteroidCollision([10,2, -5])
    sol735=sol.asteroidCollision([5,10,-5])
    sol735_1 = sol.asteroidCollision([5, -5])

    sol2352=sol.equalPairs( [[3,1,2,2],[1,4,4,5],[2,4,2,2],[2,4,2,2]]);

    sol1657=sol.closeStrings("abca","bacc")

    sol2215=sol.findDifference([1,2,3],[2,4,6])

    sol724=sol.pivotIndex([1,7,3,6,5,6])
    sol724_1 = sol.pivotIndex([-1, -1, -1, -1, -1, 0])

    sol1493=sol.longestSubarray([1,1,0,1])
    sol1493_1= sol.longestSubarray([0,1,1,1,0,1,1,0,1])
    sol1493_26 = sol.longestSubarray([1,1,0,0,1,1,1,0,1])

    sol1004_0 = sol.longestOnes([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0], 2)
    sol1004_5 = sol.longestOnes([1, 1, 1, 0, 0, 0, 1, 1, 1, 1], 0)
    sol1004=sol.longestOnes([0,1,1,0,1,1,1,1,0,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,1,1,0,1,1,1,1,0,0,1,0,1,0,1,1,0,0,0,1,0,0,1,1,0,0,0,1,1,0,1,0,1,1,0,0,1,0,0,0,0,1,1,1,1,0,0,0,1,0,1,1,0,1,0,1,1,1,1,1,1,1,1,0,1,0,1,1,1,0,0,0,1,1,1,1,1,1,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,1,1,0,0,1,0,0,1,0,1,0,0,1,1,1,0,0,0,1,1,0,1,1,1,0,0,0,1,1,1,0,1,1,0,1,1,1,0,0,1,1,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,1,1,1,1,0,0,1,1,1,1,0,1,1,0,1,0,1,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,0,0,1,1,1,1,1,0,1,1,1,1,1,1,1,0,1,0,1,0,0,1,0,1,0,0,0,1,1,1,0,1,1,0,1,1,1,1,1,0,1,0,1,0,1,0,1,1,1,0,1,0,1,1,0,1,0,1,0,0,1,0,1,0,0,0,0,1,1,1,0,1,0,1,0,0,1,1,0,1,0,0,0,0,0,1,1,0,1,0,0,0,0,1,0,0,0,0,0,1,1,1,1,1,0,1,0,1,0,0,0,0,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,1,1,1,0,1,1,1,0,0,1,0,1,1,1,0,1,1,1,1,0,1,0,0,0,0,0,1,1,1,0,0,0,0,1,1,0,0,0,0,1,1,1,0,1,0,0,0,0,0,0,0,1,1,0,1,0,1,1,1,0,0,1,0,0,0,1,0,0,1,1,1,1,1,0,0,1,1,0,0,1,0,1,1,1,1,0,0,1,1,1,0,1,1,1,0,0,1,0,0,1,0,0,1,1,1,1,0,1,0,0,0,1,1,1,1,1,0,1,0,0,1,1,0,0,1,0,1,1,1,0,1,0,0,0,0,1,1,0,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1,1,0,0,1,0,0,1,1,1,1,1,1,0,0,1,1,0,0,1,0,1,1,0,1,0,1,1,0,1,1,1,0,0,0,0,1,1,1,0,1,1,0,1,1,0,0,1,1,1,1,1,0,1,1,0,1,0,1,1,1,0,1,1,1,1,0,0,0,0,0,0,0,1,0,1,1,0,1,0,1,0,0,1,1,0,1,1,1,1,0,1,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,1,1,0,0,1,0,1,1,0,1,1,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,1,1,1,1,0,1,1,0,0,0,1,1,0,0,0,1,1,0,0,0,1,1,1,0,1,0,1,1,0,0,1,1,1,0,1,1,1,0,0,0,1,1,0,1,0,1,1,1,1,0,1,0,1,1,0,1,0,0,1,0,0,0,1,1,0,0,0,0,1,0,1,0,0,0,0,1,1,0,0,0,0,1,1,1,0,0,0,0,1,0,1,1,0,0,0,1,0,0,0,0,1,0,1,1,0,1,0,1,1,0,0,1,0,0,1,0,0,1,1,0,1,0,1,0,1,1,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,0,1,0,1,1,0,1,1,0,0,0,0,0,1,0,0,1,0,1,1,0,1,0,1,1,1,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,1,1,0,1,1,0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,0,0,1,1,1,0,1,1,1,1,0,1,0,0,1,0,0,0,1,0,0,0,1,1,0,0,1,0,0,0,0,0,1,1,0,0,0,0,1,1,1,0,0,1,0,1,0,0,1,1,1,1,1,0,1,1,1,1,1,0,0,1,0,0,0,1,1,1,0,1,0,1,1,1,0,0,1,1,1,1,1,0,0,0,0,1,1,1,0,0,0,0,1,1,1,1,1,0,1,1,1,1,1,0,1,1,0,0,1,1,1,1,1,0,0,0,1,1,1,0,0,0,1,1,0,1,0,1,0,0,0,1,1,0,1,1,0,1,0,1,1,1,1,0,0,1,1,0,1,0,0,1,0,1,1,0,1,0,1,1,1,1,0,1,0,0,1,0,1,0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,1,1,0,0,1,1,1,1,1,0,1,1,0,0,1,0,1,1,0,1,1,0,1,1,1,0,0,0,1,0,1,0,0,1,0,1,1,0,0,1,1,0,1,0,1,0,0,1,0,0,0,1,1,1,0,1,1,1,0,0,1,0,0,0,1,1,1,0,0,1,1,1,1,1,0,1,1,1,1,0,0,1,1,1,0,1,0,0,1,0,1,1,1,0,0,0,0,1,0,1,1,1,0,0,1,1,1,1,1,1,0,1,0,1,0,1,1,0,1,0,1,1,0,1,1,1,1,1,1,0,1,0,0,1,0,1,1,0,0,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,1,0,1,0,1,0,0,0,0,1,1,0,0,0,0,1,1,1,0,0,1,1,1,0],256)

    sol1456=sol.maxVowels("abciiidef", 3)

    sol392=sol.isSubsequence( "abc", "ahbgdc")

    sol283=sol.moveZeroes([0,1,0,3,12])

    kidWithCandies=sol.kidsWithCandies([2,3,5,1,3],3)
    minS=sol.minimizeStringValue("a??b?")
    maxHappiness=sol.maximumHappinessSum([1,2,3], 2)
    sol.minimumBoxes([1,3,2], [4,3,1,5,2])
    image=[[5,6,7,10],[8,9,10,10],[11,12,13,10]]
    threshold = 3
    r=sol.resultGrid(image,threshold)
