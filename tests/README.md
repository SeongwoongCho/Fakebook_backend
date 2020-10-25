# Running tests

## Requirements for running the tests

You need to run the server before run the tests.

## Test with single image file

To request single image, run ```request.py``` with ```-i``` option.

```bash
$ python3 request.py -i ${Image File Path}
```

Example

```bash
$ python3 request.py -i 'images/minseong.jpg'
{
    "key": "value"
}
```
