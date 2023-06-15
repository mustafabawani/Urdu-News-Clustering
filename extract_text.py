with open("۔کلک میں ٹيکنالوجي کي دنيا سے تازہ ترين خبروں کے ساتھ ساتھ یہ ديکھيے.doc", "r") as f:
    data = f.read()

english_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
for c in data:
    if c in english_letters:
        break

lines_before_date = data.split(c)[0].split("\n")
lines_with_content = [line for line in lines_before_date if line.strip() != ""]
second_para = "\n".join(lines_with_content[1:])
print(second_para)