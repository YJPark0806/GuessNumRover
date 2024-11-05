import numpy as np
from torchvision import datasets, transforms
from skimage.transform import resize

def preprocess_and_save_mnist():
    """
    Loads the raw MNIST dataset, selects 5000 samples for each digit,
    processes each image to a 50x50 binary matrix,
    and saves the result in a dictionary format as 'mnist_data.npy'.
    """
    # Load the MNIST dataset
    transform = transforms.ToTensor()
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Initialize a dictionary to store processed data by digit, limited to 5000 samples each
    mnist_data = {i: [] for i in range(10)}

    # Dictionary to keep track of the number of samples stored for each digit
    digit_counts = {i: 0 for i in range(10)}

    # Process and save only 5000 samples per digit
    for img, label in mnist_dataset:
        if digit_counts[label] < 100:
            # Convert the 28x28 image to a 50x50 binary matrix
            img_np = img.squeeze().numpy()  # Convert Tensor to Numpy array
            resized_img = resize(img_np, (50, 50), anti_aliasing=True)
            binary_img = (resized_img > 0.3).astype(int)  # Convert values > 0.5 to 1, others to 0

            # Append the processed image to the corresponding digit's list in the dictionary
            mnist_data[label].append(binary_img)
            digit_counts[label] += 1

        # Stop processing if all digits have 5000 samples
        if all(count == 100 for count in digit_counts.values()):
            break

    # Save the dictionary to a .npy file in the current directory
    np.save("mnist_data.npy", mnist_data)
    print("MNIST data has been processed with 5000 samples per digit and saved as 'mnist_data.npy'.")

    # Print the number of samples for each digit
    for digit in range(10):
        print(f"Number of samples for digit {digit}: {len(mnist_data[digit])}")

# Run the function
if __name__ == "__main__":
    preprocess_and_save_mnist()