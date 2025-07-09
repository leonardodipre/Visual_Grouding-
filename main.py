def main():
    try:
        num1 = float(input("Enter the first number: "))
        operation = input("Enter operation (+, *, /): ").strip()
        num2 = float(input("Enter the second number: "))
        if operation == '+':
            result = num1 + num2
        elif operation == '*':
            result = num1 * num2
        elif operation == '/':
            if num2 == 0:
                print("Error: Division by zero.")
                return
            result = num1 / num2
        else:
            print("Invalid operation.")
            return
        print("Result:", result)
    except ValueError:
        print("Invalid input. Please enter numeric values.")

if __name__ == "__main__":
    main()
