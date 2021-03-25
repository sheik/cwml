package main

import (
	"flag"
	"sync"
	"bufio"
	"os"
	"os/exec"
	"log"
)

var (
	batchFile = flag.String("filename", "", "The file holding the batch jobs")
	jobs = flag.Int("jobs", 1, "Number of parallel workers")
	wg sync.WaitGroup
)

func consumer(queue <-chan string) {
	for {
		command :=<-queue
		parts := append([]string{"-c"}, command)
		cmd := exec.Command("/bin/bash", parts...)
		log.Printf("Executing command: %s", cmd)
		err := cmd.Run()
		if err != nil {
			log.Println(err)
		}
		wg.Done()
	}
}

func main() {
	flag.Parse()
	queue := make(chan string, 10000)

	for i := 0; i < *jobs; i++ {
		go consumer(queue)
	}
	fp, err := os.Open(*batchFile)
	if err != nil {
		panic(err)
	}
	defer fp.Close()

	scanner := bufio.NewScanner(fp)
	for scanner.Scan() {
		wg.Add(1)
		queue<-scanner.Text()
	}

	if err := scanner.Err(); err != nil {
		panic(err)
	}

	wg.Wait()
	os.Exit(0)
}
