from gradio_client import Client, handle_file

client = Client("slinkp/is_it_a_cat")
result = client.predict(
        img=handle_file('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'),
        api_name="/predict"
)
print(result)
