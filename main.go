package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"

	_ "embed"

	"github.com/abhirockzz/amazon-bedrock-langchain-go/llm"
	"github.com/abhirockzz/amazon-bedrock-langchain-go/llm/claude"
	ddbhist "github.com/abhirockzz/langchaingo-dynamodb-chat-history/dynamodb_chat_history"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/memory"
	"github.com/tmc/langchaingo/outputparser"
	"github.com/tmc/langchaingo/prompts"
)

var ddbcmh *ddbhist.DynamoDBChatMessageHistory
var claudeLLM *claude.LLM
var chain chains.LLMChain
var ddbMemory *memory.ConversationBuffer

var dynamodbChatHistoryTableName string
var dynamodbChatHistoryTablePrimaryKeyName string

//go:embed static/index.html
var embedHTML string

const template = "{{.chat_history}}\n\nHuman:{{.human_input}}\n\nAssistant:"

func init() {

	var err error

	dynamodbChatHistoryTableName = os.Getenv("DYNAMODB_TABLE_NAME")
	if dynamodbChatHistoryTableName == "" {
		log.Fatal("missing env variable DYNAMODB_TABLE_NAME")
	}

	dynamodbChatHistoryTablePrimaryKeyName = os.Getenv("DYNAMODB_PRIMARY_KEY_NAME")
	if dynamodbChatHistoryTablePrimaryKeyName == "" {
		log.Fatal("missing env variable DYNAMODB_PRIMARY_KEY_NAME")
	}

	ddbcmh, err = ddbhist.New(os.Getenv("AWS_REGION"), ddbhist.WithTableName(dynamodbChatHistoryTableName), ddbhist.WithPrimaryKeyName(dynamodbChatHistoryTablePrimaryKeyName))

	if err != nil {
		log.Fatal(err)
	}

	claudeLLM, err = claude.New(os.Getenv("AWS_REGION"), llm.DontUseHumanAssistantPrompt())

	if err != nil {
		log.Fatal(err)
	}

	ddbMemory = memory.NewConversationBuffer(
		memory.WithMemoryKey("chat_history"),
		memory.WithAIPrefix("\n\nAssistant"),
		memory.WithHumanPrefix("\n\nHuman"),
	)

	chain = chains.LLMChain{
		Prompt: prompts.NewPromptTemplate(
			template,
			[]string{"chat_history", "human_input"},
		),
		LLM:          claudeLLM,
		Memory:       ddbMemory,
		OutputParser: outputparser.NewSimple(),
		OutputKey:    "text",
	}
}

func main() {
	r := gin.Default()

	r.GET("/", func(c *gin.Context) {

		chatID := uuid.NewString()
		fmt.Println("new chat session:", chatID)

		ddbcmh.PrimaryKeyValue = chatID
		ddbMemory.ChatHistory = ddbcmh
		chain.Memory = ddbMemory

		// Serve the embedded HTML file
		c.Data(http.StatusOK, "text/html", []byte(embedHTML))

	})

	r.POST("/stream", func(c *gin.Context) {

		body, err := c.GetRawData()
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read request body"})
			return
		}

		message := string(body)
		fmt.Printf("Received message: %s\n", message)

		_, err = chains.Call(c.Request.Context(), chain, map[string]any{"human_input": message}, chains.WithMaxTokens(8191), chains.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {

			fmt.Print(string(chunk))

			c.Stream(func(w io.Writer) bool {
				fmt.Fprintf(w, (string(chunk)))
				return false
			})

			return nil
		}))

		if err != nil {
			log.Fatal(err)
		}
	})

	r.Run()
}
