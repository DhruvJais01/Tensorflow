import sys
import math
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['acc'])

model.fit(X_train_scaled, y_train, epochs=5)

# Imports

# Constants
PI = 3.14159
NAME = "Neon Python"

# Function definition


def calculate_area(radius):
    if radius <= 0:
        raise ValueError("Radius must be positive.")
    area = PI * (radius ** 2)
    return area

# Class definition


class Circle:
    def __init__(self, radius):
        self.radius = radius

    def display_area(self):
        try:
            area = calculate_area(self.radius)
            print(f"Area of circle with radius {self.radius}: {area}")
        except ValueError as e:
            print(f"Error: {e}")


# Main block
if __name__ == "__main__":
    print(f"Welcome to {NAME}!")
    circle = Circle(5)
    circle.display_area()

    # Loop and conditions
    for i in range(3):
        print(f"Iteration {i+1}")
        if i % 2 == 0:
            print("Even iteration.")
        else:
            print("Odd iteration.")

    # Using math module
    print(f"Square root of 16 is {math.sqrt(16)}")

    # sys module usage
    print(f"Python version: {sys.version}")
