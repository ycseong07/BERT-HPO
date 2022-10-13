docker run \
        -d \
        -p 8080:8080 \
        --name "ml-py-gpu" \
        -v $(pwd)/workspace:/workspace \
        --env AUTHENTICATE_VIA_JUPYTER="1234" \
        --restart always \
        --gpus all \
        mltooling/ml-workspace-gpu
