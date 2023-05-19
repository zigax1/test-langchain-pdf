from flask import Flask, request, jsonify, Response, stream_with_context
from document_thread import DocumentThread
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
document_thread = DocumentThread()

@app.route("/info/<int:collection_id>", methods=["GET"])
def info(collection_id):
    response = document_thread.getInfo(collection_id)
    return jsonify(response)

@app.route("/collection/<int:collection_id>", methods=["POST"])
def add_new_thread(collection_id):
    response = document_thread.createCollection(collection_id)
    return jsonify(response)

@app.route("/collection/<int:collection_id>/load_document", methods=["POST"])
def load_document(collection_id):
    fileName = request.form["fileName"]
    bucketName = request.form["bucketName"]

    # if not threads_handler.checkIdExists(collection_id):
    #     return jsonify({"message": "Thread not found", "id": collection_id})
    document_thread.loadFile(collection_id, fileName, bucketName)
    return jsonify({"message": "Document loaded"})

def stream(collection_id, question):
    completion = document_thread.askQuestion(collection_id, question)
    for line in completion:
        yield 'data: %s\n\n' % line

@app.route("/collection/<int:collection_id>/ask_question", methods=["POST"])
def ask_question(collection_id):
    question = request.form["question"]
    # response_generator = document_thread.askQuestion(collection_id, question)
    # return jsonify(response_generator)
    def stream():
        completion = document_thread.askQuestion(collection_id, question)
        print(type(completion))
        print(completion)
        for line in completion['answer']:
            yield line

    return Response(stream(), mimetype='text/event-stream')

    # return Response(response_generator, mimetype="application/json")

    # def stream():
    #     completion = document_thread.askQuestion(collection_id, question)
    #     for line in completion:
    #         yield 'data: %s\n\n' % line

    # return Response(stream(), mimetype='text/event-stream')




# # Create an S3 client
# s3_client = boto3.client(
#     's3',
#     aws_access_key_id=AWS_ACCESS_KEY_ID,
#     aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#     region_name=S3_REGION_NAME
# )


# @app.route('/uploadToS3', methods=['POST'])
# def upload_file():
#     file = request.files['file']

#     # Check if the file is empty
#     if file.filename == '':
#         return 'Empty filename', 400

#     # Upload the file to S3
#     try:
#         s3_client.create_bucket(Bucket=S3_BUCKET_NAME, CreateBucketConfiguration={
#             'LocationConstraint': S3_REGION_NAME
#         })
#         s3_client.upload_fileobj(file, S3_BUCKET_NAME, file.filename)
#     except Exception as e:
#         return str(e), 500

#     return 'File uploaded successfully'

# @app.route('/getFromS3/<filename>', methods=['GET'])
# def get_file_url(filename):
#     # Get the URL of the uploaded file
#     file_url = s3_client.generate_presigned_url(
#         'get_object',
#         Params={'Bucket': S3_BUCKET_NAME, 'Key': filename},
#         ExpiresIn=36000  # URL expiration time in seconds
#     )

#     return file_url

if __name__ == "__main__":
    app.run(debug=True, port=3000)