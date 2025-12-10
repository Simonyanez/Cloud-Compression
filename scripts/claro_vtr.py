from typing import List
import traceback

def divisibleSumPairs(array: List, k: int):
    N = len(array)
    assert k>0, "K must be a positive integer"
    assert N > 1 and N < 101, "Array length out of range"
    assert k < 101, "K out of range"
    assert max(array) < 101 and min(array) > 0, "Array values out of range"
    i_vals = []
    j_vals = []
    for i_idx,i in enumerate(array):
        for j_idx, j in enumerate(array):
            if (i+j) % k == 0 and i not in j_vals and i!=j:
                i_vals.append(i)
                j_vals.append(j)
                print(f"Tuple positions:{i_idx,j_idx} and values: {i,j} added to list")

    return len(i_vals)

def divisibleSumPairbyClaude(array, k):
    """
    Find pairs in array where the sum is divisible by k.
    
    Args:
        array: List of positive integers
        k: Divisor (positive integer)
    
    Returns:
        int: Number of pairs whose sum is divisible by k
    """
    
    # print(f"DEBUG: Function called with array={array}, k={k}")
    
    # Restriction 1: Array length should be between 2 and 100 (inclusive)
    # print(f"DEBUG: Checking array length: {len(array)}")
    assert 2 <= len(array) <= 100, f"Array length must be between 2 and 100, got {len(array)}"
    # print("DEBUG: Array length check passed ✓")
    
    # Restriction 2: Divisor k should be between 1 and 100 (inclusive)
    # print(f"DEBUG: Checking divisor k: {k}")
    assert 1 <= k <= 100, f"Divisor k must be between 1 and 100, got {k}"
    # print("DEBUG: Divisor k check passed ✓")
    
    # Restriction 3: All array values should be between 1 and 100 (inclusive)
    # print(f"DEBUG: Checking array values...")
    for i, value in enumerate(array):
        # print(f"DEBUG: Checking array[{i}] = {value}")
        assert 1 <= value <= 100, f"Array value at index {i} must be between 1 and 100, got {value}"
    # print("DEBUG: All array values check passed ✓")
    
    # print("DEBUG: All restrictions satisfied, proceeding with pair counting...")
    
    # Count pairs whose sum is divisible by k
    pair_count = 0
    pairs_found = []
    
    # print("DEBUG: Starting pair analysis...")
    for i in range(len(array)):
        for j in range(i + 1, len(array)):
            current_sum = array[i] + array[j]
            is_divisible = current_sum % k == 0
            
            # print(f"DEBUG: Pair ({i},{j}): array[{i}]={array[i]}, array[{j}]={array[j]}, sum={current_sum}, divisible by {k}? {is_divisible}")
            
            if is_divisible:
                pair_count += 1
                pairs_found.append((i, j, array[i], array[j], current_sum))
                # print(f"DEBUG: ✓ Valid pair found! Count is now {pair_count}")
    
    # print(f"\nDEBUG: Summary of valid pairs found:")
    for pair in pairs_found:
        i, j, val1, val2, sum_val = pair
        # print(f"DEBUG: Indices ({i},{j}): {val1} + {val2} = {sum_val} (divisible by {k})")
    
    # print(f"DEBUG: Total pairs found: {pair_count}")
    return pair_count

def test_by_cloud(test_array, test_k):
    
    print(f"Test case: array = {test_array}, k = {test_k}")
    print("-" * 60)
    
    try:
        print(f"This is the Claude TEST")
        result = divisibleSumPairbyClaude(test_array, test_k)
        print(f"\n🎉 RESULT: {result} pairs found whose sum is divisible by {test_k}")
        
        print(f"This is my own function")
        result = divisibleSumPairs(test_array, test_k)
        print(f"\n🎉 RESULT: {result} pairs found whose sum is divisible by {test_k}")
        
    except AssertionError as e:
        print(f"❌ ASSERTION ERROR: {e}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
    
    print("="*60)
    print("TEST COMPLETED")
    print("="*60)
if __name__ == "__main__":
    print("="*60)
    print("TESTING divisibleSumPair FUNCTION")
    print("="*60)
    
    # Test case
    test_array = [1, 3, 2, 6, 1, 2]
    test_k = 3

    # Test case 2
    test_array2 = [1, 2, 3, 4, 5, 6]
    test_k2 = 5
    test_by_cloud(test_array, test_k)
    test_by_cloud(test_array2, test_k2)

    # ar = [1000+i for i in range(98)]
    # k = 5
    # try:
    #     count = divisibleSumPairs(ar, k)
    #     print(f"Total found tuples: {count}")
    # except Exception as e:
    #     print(f"Captured error: {e} \n {traceback.format_exc()}")
    #
    # pass
