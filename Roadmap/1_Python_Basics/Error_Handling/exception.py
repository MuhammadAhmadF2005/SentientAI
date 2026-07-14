def exception_handling():
    try:
        number = int(input("Enter a divisor: "))
        result = 10 / number
    except ZeroDivisionError:
        print("Cannot divide by zero")
    except ValueError:
        print("Invalid input")
    else:
        print("Result is ", result)
    finally:
        print("Program finished") 

###now we define our own errorType using OOP

class InsufficientFundsError(Exception):
    pass

def withdraw(amount, balance):
    try:
        if amount > balance:
            raise InsufficientFundsError("insufficient funds")
        balance = balance - amount
        print(f"New balance: {balance}")
    except InsufficientFundsError as e:
        print(e)
    else:
        print("withdrawal successful")

withdraw(100, 500)

