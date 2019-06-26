package proving

import (
	"fmt"
	"github.com/spacemeshos/merkle-tree"
	"github.com/spacemeshos/merkle-tree/cache"
	"github.com/spacemeshos/post/persistence"
	"github.com/spacemeshos/post/shared"
	"io"
	"math"
	"runtime"
)

const (
	LabelGroupSize = shared.LabelGroupSize
)

var (
	VerifyInitialized = shared.VerifyInitialized
)

type (
	Config          = shared.Config
	Logger          = shared.Logger
	Difficulty      = shared.Difficulty
	Challenge       = shared.Challenge
	CacheReader     = cache.CacheReader
	LayerReadWriter = cache.LayerReadWriter
)

type Prover struct {
	cfg    *Config
	logger Logger
}

func NewProver(cfg *Config, logger Logger) *Prover { return &Prover{cfg, logger} }

func (p *Prover) GenerateProof(id []byte, challenge Challenge) (proof *Proof,
	err error) {
	proof, err = p.generateProof(id, challenge)
	if err != nil {
		err = fmt.Errorf("proof generation failed: %v", err)
		p.logger.Error(err.Error())
	}
	return proof, err
}

func (p *Prover) generateProof(id []byte, challenge Challenge) (*Proof, error) {
	if err := VerifyInitialized(p.cfg, id); err != nil {
		return nil, err
	}

	difficulty := Difficulty(p.cfg.Difficulty)
	if err := difficulty.Validate(); err != nil {
		return nil, err
	}

	proof := new(Proof)
	proof.Challenge = challenge
	proof.Identity = id

	result, err := p.exec(id, challenge, difficulty)
	if err != nil {
		return nil, err
	}

	leafReader := result.reader.GetLayerReader(0)
	width, err := leafReader.Width()
	if err != nil {
		return nil, err
	}

	proof.MerkleRoot = result.root
	provenLeafIndices := CalcProvenLeafIndices(
		proof.MerkleRoot, width<<difficulty, uint8(p.cfg.NumOfProvenLabels), difficulty)

	_, proof.ProvenLeaves, proof.ProofNodes, err = merkle.GenerateProof(provenLeafIndices, result.reader)
	if err != nil {
		return nil, err
	}

	err = leafReader.Close()
	if err != nil {
		return nil, err
	}

	return proof, nil
}

type execResult struct {
	reader CacheReader
	root   []byte
}

type execFileResult struct {
	index int
	*execResult
}

func (p *Prover) exec(id []byte, challenge Challenge, difficulty Difficulty) (*execResult, error) {
	dir := shared.GetInitDir(p.cfg.DataDir, id)
	readers, err := persistence.GetReaders(dir)
	if err != nil {
		return nil, err
	}

	if len(readers) > 1 && p.cfg.MaxReadParallelism > 1 {
		results, err := p.execParallel(readers, challenge)
		if err != nil {
			return nil, err

		}
		return merge(results)
	} else {
		reader, err := persistence.Merge(readers)
		if err != nil {
			return nil, err
		}

		width, err := reader.Width()
		if err != nil {
			return nil, err
		}
		if width*difficulty.LabelsPerGroup() >= math.MaxUint64 {
			return nil, fmt.Errorf("leaf reader too big, number of label groups (%d) * labels per group (%d) "+
				"overflows uint64", width, difficulty.LabelsPerGroup())
		}

		return p.execFile(reader, challenge)
	}
}

func (p *Prover) execFile(reader LayerReadWriter, challenge Challenge) (*execResult, error) {
	cacheWriter := cache.NewWriter(cache.MinHeightPolicy(p.cfg.LowestLayerToCacheDuringProofGeneration),
		cache.MakeSliceReadWriterFactory())

	tree, err := merkle.NewTreeBuilder().WithHashFunc(challenge.GetSha256Parent).WithCacheWriter(cacheWriter).Build()
	if err != nil {
		return nil, err
	}

	for {
		leaf, err := reader.ReadNext()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		err = tree.AddLeaf(leaf)
		if err != nil {
			return nil, err
		}
	}

	cacheWriter.SetLayer(0, reader)
	cacheReader, err := cacheWriter.GetReader()

	return &execResult{
		reader: cacheReader,
		root:   tree.Root(),
	}, nil
}

func (p *Prover) execParallel(readers []LayerReadWriter, challenge Challenge) ([]*execResult, error) {
	numOfWorkers := p.CalcParallelism(runtime.NumCPU())
	jobsChan := make(chan int, len(readers))
	resultsChan := make(chan *execFileResult, len(readers))
	errChan := make(chan error, 0)

	p.logger.Info("execution: start executing %v files, parallelism degree: %v", len(readers), numOfWorkers)

	for i := 0; i < len(readers); i++ {
		jobsChan <- i
	}
	close(jobsChan)

	for i := 0; i < numOfWorkers; i++ {
		go func() {
			for {
				index, more := <-jobsChan
				if !more {
					return
				}

				res, err := p.execFile(readers[index], challenge)
				if err != nil {
					errChan <- err
					return
				}

				resultsChan <- &execFileResult{index, res}
			}
		}()
	}

	results := make([]*execResult, len(readers))
	for i := 0; i < len(readers); i++ {
		select {
		case res := <-resultsChan:
			results[res.index] = res.execResult
		case err := <-errChan:
			return nil, err
		}
	}

	return results, nil
}

func merge(results []*execResult) (*execResult, error) {
	switch len(results) {
	case 0:
		return nil, nil
	case 1:
		return results[0], nil
	default:
		readers := make([]CacheReader, len(results))
		for i, result := range results {
			readers[i] = result.reader
		}

		reader, err := cache.Merge(readers)
		if err != nil {
			return nil, err
		}

		reader, root, err := cache.BuildTop(reader)
		if err != nil {
			return nil, err
		}

		return &execResult{reader, root}, nil
	}
}

func (p *Prover) CalcParallelism(maxParallelism int) int {
	maxParallelism = max(maxParallelism, 1)
	readParallelism := max(int(p.cfg.MaxReadParallelism), 1)
	return min(readParallelism, maxParallelism)
}

func min(x, y int) int {
	if x < y {
		return x
	}
	return y
}

func max(x, y int) int {
	if x > y {
		return x
	}
	return y
}
