## FastAPI

## Small FastAPI example below (e.g. to-do list)

# TODO: build/deploy FastAPI endpoint for the GPT Streamlit App to consume rather than the OpenAI API.

from fastapi import FastAPI
from models import Todo

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


todos = []


# get all todos
@app.get("/todos")
async def get_todos():
    return {"todos": todos}


# get a single todo
@app.get("/todos/{todo_id}")
async def get_todo(todo_id: int):
    todo_list = [x for x in todos if x.id == todo_id]
    if len(todo_list) == 0:
        return {"message": "No todos found"}
    else:
        return todo_list[0]


# create a todo
@app.post("/todos")
async def create_todos(todo: Todo):
    todos.append(todo)
    return {"message": "Todo has been added"}


# update a todo
@app.put("/todos/{todo_id}")
async def update_todos(todo_id: int, todo_obj: Todo):
    todo_list = [x for x in todos if x.id == todo_id]
    if len(todo_list) > 0:
        for t in todo_list:
            t.id = todo_id
            t.item = todo_obj.item
        return {"message": "Todo has been updated"}
    else:
        return {"message": "No such todo.."}


# delete a todo
@app.delete("/todos/{todo_id}")
async def get_todo(todo_id: int):
    todo_list = [x for x in todos if x.id == todo_id]
    if len(todo_list) > 0:
        todos.remove(todo_list[0])
        return {"message": "todo deleted"}
    else:
        return {"message": "No todos found"}
