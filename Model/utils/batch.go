package main

import (
	"fmt"
	"flag"
	"sync"
	"bufio"
	"os"
	"os/exec"
	"log"
)

var (
	batchFile = flag.String("f", "", "The file holding the batch jobs")
	jobs = flag.Int("j", 2, "Number of parallel workers")
	wg sync.WaitGroup
)

// Consumer code
func consumer(queue <-chan string) {
	for {
		command :=<-queue
		parts := []string{"-c", command}
		cmd := exec.Command("/bin/bash", parts...)
		log.Printf("Executing command: %s", cmd)
		err := cmd.Run()
		if err != nil {
			log.Println("Command failed:", cmd, ":", err)
		}
		wg.Done()
	}
}

func main() {
	// Parse flags
	flag.Parse()
	if *batchFile == "" {
		fmt.Printf("Usage of %s:\n", os.Args[0])
		flag.PrintDefaults()
		os.Exit(0)
	}
	if *jobs < 1 {
		log.Fatal("Jobs (-j) should be greater than 0")
	}

	// Queue for feeding commands to consumers
	queue := make(chan string, 10000)

	// Start up consumer goroutines
	// One for each number of jobs
	for i := 0; i < *jobs; i++ {
		go consumer(queue)
	}

	// Open file for reading
	fp, err := os.Open(*batchFile)
	if err != nil {
		log.Fatal(err)
	}
	defer fp.Close()

	// Producer code
	// Read file line by line and feed commands
	// to the consumer goroutines
	scanner := bufio.NewScanner(fp)
	for scanner.Scan() {
		wg.Add(1)
		queue<-scanner.Text()
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}

	// Wait for goroutines to finish work
	wg.Wait()
	os.Exit(0)
}
