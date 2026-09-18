import sqlite3

connection = sqlite3.connect("data/translator.sqlite3")
for row in connection.execute(
    "select id,status,stage,progress,review_count,error,updated_at "
    "from jobs order by updated_at desc limit 10"
):
    print(row)
