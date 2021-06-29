from ProcessImage import ProcessImage

filesAddress = input("Enter your image file path (directory or single file): ")

top = int(input("Enter total pixel should crop from (top) (Default 0): ") or 0)  # 285
right = int(
    input("Enter total pixel should crop from (right) (Default 0): ") or 0)  # 20
bottom = int(
    input("Enter total pixel should crop from (bottom) (Default 0): ") or 0)  # 230
left = int(
    input("Enter total pixel should crop from (left) (Default 0): ") or 0)  # 20

processor = ProcessImage(filesAddress, top, right, bottom, left)

processor.processFiles()
