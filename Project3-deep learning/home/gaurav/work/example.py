def convolve(input_pixels, kernel):
    """
    Convolves the input_pixels with kernel (flipping the kernel) and returns the value at position (2,2).

    :param input_pixels: 2D list of integers representing the input image pixels.
    :param kernel: 2D list of integers representing the kernel.
    :return: Integer representing the convolved value at position (2,2).
    """

  # Flip the kernel vertically and horizontally
    flipped_kernel = [row[::-1] for row in kernel[::-1]]

    # Initialize the output value
    output_value = 0

    # Perform convolution at position (2,2)
    for i in range(-1, 2):
        for j in range(-1, 2):
            output_value += input_pixels[1 + i][1 + j] * flipped_kernel[1 + i][1 + j]

    return int(output_value)

# Given input_pixels and kernel
# input_pixels = [
#     [105, 102, 100],
#     [103, 99, 103],
#     [101, 98, 104]
# ]

input_pixels = [
    [1, 2, 3, 4, 5],
    [5, 6, 7, 8, 9],
    [9, 10, 11, 12, 13],
    [13, 14, 15, 16, 17],
    [17, 18, 19, 20, 21]
]

kernel = [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
]

result = convolve(input_pixels, kernel)
print(result)  # This will output the convolved value at position (2,2) with the kernel flipped
