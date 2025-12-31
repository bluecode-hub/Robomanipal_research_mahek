# Complete flow of the working

## Prompt separation
A different file has been created called prompts which handles all the promts. changing the prompts become easier it does not involve any change the sructure of the code just the prompts file is enough.

## Output validity
the code is written in such a way that it is restricted to sending responses only in the form of JSON file. the outputs are parsed and checked if it is returning in the format that we want. if that is not followed errors are also raised.

## Failure Handling

if the validation fails in the above that time the request is retires and the errors are logged and shown to the use. the system fails after retiring. this is included in the parser.py file and the logger.py file.

## Logging
all the logs are recorded and it records the prompts given the model response the error. every single step is recorded with the time the event and the message.

## Limitation

the system does not attempt output repair. 
