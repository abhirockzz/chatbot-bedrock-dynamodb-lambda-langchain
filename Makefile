build-ChatFunction:
	GOARCH=amd64 GOOS=linux CGO_ENABLED=0 go build -o bootstrap .
	mv bootstrap $(ARTIFACTS_DIR)/bootstrap