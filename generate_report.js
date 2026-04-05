const fs = require("fs");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageNumber, ExternalHyperlink, ImageRun, TabStopType,
  TabStopPosition, LevelFormat,
} = require("docx");

const logoBuaa = fs.readFileSync("university logo.png");
const logoRcssteap = fs.readFileSync("RCSSTEAP.jpg");

// ── Design tokens (matching Assignment 2 report) ──
const NAVY = "00337F";
const BODY = "333333";
const META = "666666";
const ALT_ROW = "F2F2F2";
const FONT = "Times New Roman";
const LINK_BLUE = "0563C1";

const STREAMLIT_URL = "https://gnss-rag-assistant-gxgbm7dbgfmsxkuycfabmt.streamlit.app/";
const GITHUB_URL = "https://github.com/Tevin-Wills/gnss-rag-assistant";

const border = { style: BorderStyle.SINGLE, size: 4, color: "B0B0B0" };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0 };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };
const cm = { top: 60, bottom: 60, left: 100, right: 100 };

// Helper: table cell
function tc(text, w, opts = {}) {
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
    margins: cm, verticalAlign: opts.vAlign || undefined,
    children: [new Paragraph({
      alignment: opts.align || AlignmentType.LEFT,
      spacing: { after: 0 },
      children: [new TextRun({
        text, font: FONT, size: opts.sz || 22,
        bold: !!opts.bold, italics: !!opts.italics,
        color: opts.color || BODY,
      })],
    })],
  });
}

// Helper: body paragraph
function bodyP(children, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after || 120, before: opts.before || 0, line: 276 },
    alignment: opts.align || AlignmentType.JUSTIFIED,
    indent: opts.indent ? { left: opts.indent } : undefined,
    numbering: opts.bullet ? { reference: "bullets", level: 0 } : undefined,
    children: Array.isArray(children) ? children : [new TextRun({ text: children, font: FONT, size: 22, color: BODY })],
  });
}

// Helper: section heading
function heading(num, title) {
  return new Paragraph({
    spacing: { before: 280, after: 120 },
    children: [new TextRun({ text: `${num}. ${title}`, font: FONT, size: 26, bold: true, color: NAVY })],
  });
}

// Helper: horizontal rule
function hr() {
  return new Paragraph({
    spacing: { before: 40, after: 40 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: NAVY, space: 4 } },
    children: [new TextRun({ text: "", size: 4 })],
  });
}

const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "\u2022",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    }],
  },
  styles: {
    default: { document: { run: { font: FONT, size: 22, color: BODY } } },
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, right: 1200, bottom: 1080, left: 1200 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          children: [
            new TextRun({ text: "Group 14 \u2014 Assignment 3 AI Process Log", font: FONT, size: 16, color: META }),
            new TextRun({ text: "\tAI & Large Models | Spring 2025", font: FONT, size: 16, color: META }),
          ],
          tabStops: [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }],
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "CCCCCC", space: 4 } },
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Page ", font: FONT, size: 16, color: META }),
            new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 16, color: META }),
            new TextRun({ text: " of ", font: FONT, size: 16, color: META }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], font: FONT, size: 16, color: META }),
          ],
        })],
      }),
    },
    children: [
      // ══════════════════════════════════════════════
      // TITLE BLOCK — logos + university
      // ══════════════════════════════════════════════
      hr(),
      new Table({
        width: { size: 9840, type: WidthType.DXA },
        columnWidths: [1800, 6240, 1800],
        rows: [new TableRow({
          children: [
            new TableCell({ borders: noBorders, width: { size: 1800, type: WidthType.DXA }, verticalAlign: "center", margins: cm,
              children: [new Paragraph({ children: [new ImageRun({ type: "png", data: logoBuaa, transformation: { width: 65, height: 65 }, altText: { title: "BUAA", description: "Beihang University", name: "buaa" } })] })],
            }),
            new TableCell({ borders: noBorders, width: { size: 6240, type: WidthType.DXA }, verticalAlign: "center", margins: cm,
              children: [
                new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: "BEIHANG UNIVERSITY", bold: true, size: 26, font: FONT, color: NAVY })] }),
                new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: "Regional Centre for Space Science and Technology Education", size: 19, font: FONT, color: META })] }),
                new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: "in Asia and the Pacific (China) \u2014 RCSSTEAP", size: 19, font: FONT, color: META })] }),
              ],
            }),
            new TableCell({ borders: noBorders, width: { size: 1800, type: WidthType.DXA }, verticalAlign: "center", margins: cm,
              children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new ImageRun({ type: "jpg", data: logoRcssteap, transformation: { width: 65, height: 65 }, altText: { title: "RCSSTEAP", description: "RCSSTEAP", name: "rcssteap" } })] })],
            }),
          ],
        })],
      }),
      hr(),

      // Assignment title box
      new Table({
        width: { size: 9840, type: WidthType.DXA },
        columnWidths: [9840],
        rows: [new TableRow({
          children: [new TableCell({
            borders, width: { size: 9840, type: WidthType.DXA },
            shading: { fill: NAVY, type: ShadingType.CLEAR },
            margins: { top: 140, bottom: 140, left: 200, right: 200 },
            children: [
              new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "ASSIGNMENT 3 \u2014 GROUP AI PROCESS LOG", font: FONT, size: 24, bold: true, color: "FFFFFF" })] }),
              new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 }, children: [new TextRun({ text: "The RAG Concept Demo", font: FONT, size: 22, color: "BBCCDD" })] }),
              new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: "GNSS Technical Knowledge Assistant using", font: FONT, size: 20, color: "FFFFFF" })] }),
              new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: [new TextRun({ text: "Retrieval-Augmented Generation", font: FONT, size: 20, color: "FFFFFF" })] }),
            ],
          })],
        })],
      }),

      // Course info
      new Table({
        width: { size: 9840, type: WidthType.DXA },
        columnWidths: [2400, 7440],
        rows: [
          ["Course:", "Artificial Intelligence and Advanced Large Models"],
          ["Program:", "Master\u2019s Degree \u2014 Beihang University / RCSSTEAP"],
          ["Semester:", "Spring 2025"],
        ].map(r => new TableRow({
          children: [
            tc(r[0], 2400, { bold: true, color: NAVY, align: AlignmentType.RIGHT, sz: 21 }),
            tc(r[1], 7440, { color: BODY, sz: 21 }),
          ],
        })),
      }),

      // Dashboard link box
      new Paragraph({ spacing: { before: 200 } , children: [] }),
      new Table({
        width: { size: 9840, type: WidthType.DXA },
        columnWidths: [9840],
        rows: [new TableRow({
          children: [new TableCell({
            borders, width: { size: 9840, type: WidthType.DXA },
            shading: { fill: NAVY, type: ShadingType.CLEAR },
            margins: { top: 100, bottom: 100, left: 200, right: 200 },
            children: [
              new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 40 }, children: [new TextRun({ text: "\u25B6 Interactive Dashboard", font: FONT, size: 22, bold: true, color: "FFFFFF" })] }),
              new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 40 }, children: [
                new ExternalHyperlink({ children: [new TextRun({ text: STREAMLIT_URL, font: FONT, size: 21, color: "BBCCDD", underline: {} })], link: STREAMLIT_URL }),
              ]}),
              new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 0 }, children: [
                new TextRun({ text: "Source Code: ", font: FONT, size: 20, bold: true, color: "FFFFFF" }),
                new ExternalHyperlink({ children: [new TextRun({ text: GITHUB_URL, font: FONT, size: 20, color: "BBCCDD", underline: {} })], link: GITHUB_URL }),
              ]}),
            ],
          })],
        })],
      }),

      // ══════════════════════════════════════════════
      // 1. GROUP INFORMATION
      // ══════════════════════════════════════════════
      heading("1", "Group Information"),
      bodyP("This assignment was completed collaboratively by Group 14, comprising three Master\u2019s students at Beihang University / RCSSTEAP. Topic selection, AI tool usage, prompt refinement, output evaluation, and final artifact selection were completed as a group."),

      new Table({
        width: { size: 7200, type: WidthType.DXA },
        columnWidths: [4200, 3000],
        rows: [
          new TableRow({ children: [
            tc("Full Name", 4200, { bold: true, color: "FFFFFF", fill: NAVY, align: AlignmentType.CENTER }),
            tc("Student ID", 3000, { bold: true, color: "FFFFFF", fill: NAVY, align: AlignmentType.CENTER }),
          ]}),
          ...[
            ["Granny Tlou Molokomme", "LS2525256"],
            ["Letsoalo Maile", "LS2525231"],
            ["Lemalasia Tevin Muchera", "LS2525229"],
          ].map((r, i) => new TableRow({ children: [
            tc(r[0], 4200, { fill: i % 2 === 0 ? ALT_ROW : "FFFFFF", align: AlignmentType.CENTER }),
            tc(r[1], 3000, { fill: i % 2 === 0 ? ALT_ROW : "FFFFFF", align: AlignmentType.CENTER }),
          ]})),
        ],
      }),

      // ══════════════════════════════════════════════
      // 2. ASSIGNMENT OBJECTIVE
      // ══════════════════════════════════════════════
      heading("2", "Assignment Objective"),
      bodyP("The objective of this assignment was to design a rapid prototype demonstrating a complete Retrieval-Augmented Generation (RAG) pipeline that connects a foundation model to domain-specific GNSS engineering knowledge. The demo addresses the core problem identified in Sessions 6\u20138: foundation models are prone to hallucination, and RAG is the engineering solution to ground model outputs in retrieved evidence. The deliverable is a working Streamlit application that ingests technical documents, chunks and embeds them into a vector database, retrieves relevant context for user queries, and generates precise, source-cited answers without hallucinating."),

      // ══════════════════════════════════════════════
      // 3. AI WORKFLOW SUMMARY
      // ══════════════════════════════════════════════
      heading("3", "AI Workflow Summary"),
      bodyP("The group followed a structured, iterative workflow to move from course concepts to polished deliverables:"),

      bodyP([
        new TextRun({ text: "Phase 1 \u2014 Brief Analysis: ", font: FONT, size: 22, bold: true, color: BODY }),
        new TextRun({ text: "Reviewed the Assignment 3 brief and identified the core requirements: feed engineering documents into an AI coding agent and design a prototype demonstrating chunking, embedding, and retrieval strategies.", font: FONT, size: 22, color: BODY }),
      ], { bullet: true }),

      bodyP([
        new TextRun({ text: "Phase 2 \u2014 Course Alignment: ", font: FONT, size: 22, bold: true, color: BODY }),
        new TextRun({ text: "Extracted and adapted key concepts from Session 6 (Transformer architectures, foundation models, hallucination), Session 7 (post-training alignment, system prompts, safety), and Session 8 (RAG pipeline architecture, chunking strategies, embedding models, vector databases, cosine similarity retrieval) to the GNSS technical knowledge domain.", font: FONT, size: 22, color: BODY }),
      ], { bullet: true }),

      bodyP([
        new TextRun({ text: "Phase 3 \u2014 Pipeline Development: ", font: FONT, size: 22, bold: true, color: BODY }),
        new TextRun({ text: "Used Claude Code to implement the complete RAG pipeline: PDF ingestion with pdfplumber, three chunking strategies (fixed-size, sentence-based, semantic), embedding with all-MiniLM-L6-v2, ChromaDB vector storage, cosine similarity retrieval, and Groq-hosted Llama 3.3 70B generation with source citation enforcement. Outputs were iteratively reviewed and refined at each stage.", font: FONT, size: 22, color: BODY }),
      ], { bullet: true }),

      bodyP([
        new TextRun({ text: "Phase 4 \u2014 UI & Evaluation: ", font: FONT, size: 22, bold: true, color: BODY }),
        new TextRun({ text: "Built a polished Streamlit interface with a dark navy + cyan theme matching the group\u2019s Assignment 2 dashboard, animated GNSS constellation sidebar, interactive pipeline diagram, Q&A tab with comparison mode (RAG vs. Plain LLM), an evaluation dashboard that benchmarks all six test scenarios with IR metrics (Hit Rate @k, MRR), and a Strategy Comparison tab comparing three chunking strategies and three chunk sizes (200, 512, 1000 tokens) with interactive Plotly charts.", font: FONT, size: 22, color: BODY }),
      ], { bullet: true }),

      bodyP([
        new TextRun({ text: "Phase 5 \u2014 Deployment: ", font: FONT, size: 22, bold: true, color: BODY }),
        new TextRun({ text: "Created a GitHub repository, pushed the codebase with the pre-built vector database, and deployed the application to Streamlit Cloud for public access.", font: FONT, size: 22, color: BODY }),
      ], { bullet: true }),

      // ══════════════════════════════════════════════
      // 4. AI TOOLS USED
      // ══════════════════════════════════════════════
      heading("4", "AI Tools Used"),

      new Table({
        width: { size: 9840, type: WidthType.DXA },
        columnWidths: [1800, 2200, 5840],
        rows: [
          new TableRow({ children: [
            tc("AI Tool", 1800, { bold: true, color: "FFFFFF", fill: NAVY, align: AlignmentType.CENTER }),
            tc("Purpose", 2200, { bold: true, color: "FFFFFF", fill: NAVY, align: AlignmentType.CENTER }),
            tc("How It Was Used", 5840, { bold: true, color: "FFFFFF", fill: NAVY, align: AlignmentType.CENTER }),
          ]}),
          ...[
            ["Claude Code", "Full pipeline development & deployment", "Built the complete RAG pipeline (ingestion, chunking, embedding, retrieval, generation); implemented three chunking strategies and three chunk-size variants; designed the Streamlit UI with dark-themed CSS, animated sidebar, pipeline diagram, and interactive Plotly charts; created the evaluation dashboard with IR metrics (Hit Rate @k, MRR) and strategy/chunk-size comparison tabs; debugged Windows-specific issues; created the GitHub repository and deployed to Streamlit Cloud."],
            ["Groq API", "LLM inference (free tier)", "Hosted the Llama 3.3 70B Versatile model for answer generation via an OpenAI-compatible API endpoint. Used for both RAG-grounded answers and plain LLM comparison mode."],
            ["ChromaDB", "Vector database", "Stored embedded document chunks with metadata (document name, page numbers) in persistent local collections. Separate collections maintained for each chunking strategy."],
            ["Sentence-Transformers", "Embedding model", "all-MiniLM-L6-v2 model used to encode document chunks and user queries into 384-dimensional dense vectors for cosine similarity retrieval."],
          ].map((r, i) => new TableRow({ children: [
            tc(r[0], 1800, { bold: true, fill: i % 2 === 0 ? ALT_ROW : "FFFFFF" }),
            tc(r[1], 2200, { fill: i % 2 === 0 ? ALT_ROW : "FFFFFF" }),
            tc(r[2], 5840, { fill: i % 2 === 0 ? ALT_ROW : "FFFFFF", sz: 20 }),
          ]})),
        ],
      }),

      // ══════════════════════════════════════════════
      // 5. SOURCE INPUTS
      // ══════════════════════════════════════════════
      heading("5", "Source Inputs"),
      bodyP("The following course materials and engineering documents were used as inputs to the AI-assisted workflow:"),

      bodyP([new TextRun({ text: "Assignment 3 instructions and evaluation criteria", font: FONT, size: 22, color: BODY })], { bullet: true }),
      bodyP([new TextRun({ text: "Session 6 \u2014 Transformer architecture, foundation models, hallucination causes and mitigation, emergent capabilities", font: FONT, size: 22, color: BODY })], { bullet: true }),
      bodyP([new TextRun({ text: "Session 7 \u2014 Post-training alignment (SFT, RLHF, DPO), system prompts, safety constraints, Constitutional AI", font: FONT, size: 22, color: BODY })], { bullet: true }),
      bodyP([new TextRun({ text: "Session 8 \u2014 RAG pipeline architecture, chunking strategies, embedding models, vector databases, cosine similarity retrieval, grounded generation", font: FONT, size: 22, color: BODY })], { bullet: true }),
      bodyP([new TextRun({ text: "Five GNSS engineering documents: GPS SPS Performance Standard, 3D Mapping-Aided GNSS (Groves), Federal Radionavigation Plan 2021, ZED-F9P Integration Manual, Urban Positioning 3DMA GNSS", font: FONT, size: 22, color: BODY })], { bullet: true }),
      bodyP([new TextRun({ text: "Group\u2019s GNSS domain knowledge \u2014 satellite positioning, multipath, signal degradation, urban canyon effects, RTK surveying", font: FONT, size: 22, color: BODY })], { bullet: true }),

      // ══════════════════════════════════════════════
      // 6. GROUP REVIEW AND REFINEMENT
      // ══════════════════════════════════════════════
      heading("6", "Group Review and Refinement Process"),
      bodyP("All AI-generated outputs were reviewed collaboratively by the group before inclusion in the final submission. The review process verified that the final materials clearly addressed:"),

      bodyP([new TextRun({ text: "RAG pipeline completeness \u2014 full document-to-answer pipeline with PDF ingestion, chunking, embedding, vector storage, retrieval, and grounded generation", font: FONT, size: 22, color: BODY })], { bullet: true }),
      bodyP([new TextRun({ text: "Hallucination mitigation \u2014 system prompt enforces source-only answers with [Source: Document, pp. X\u2013Y] citations; comparison mode demonstrates RAG vs. plain LLM", font: FONT, size: 22, color: BODY })], { bullet: true }),
      bodyP([new TextRun({ text: "Multiple chunking strategies \u2014 fixed-size (676 chunks), sentence-based (1,515 chunks), and semantic similarity (2,880 chunks) stored in separate collections", font: FONT, size: 22, color: BODY })], { bullet: true }),
      bodyP([new TextRun({ text: "Chunk-size comparison \u2014 three fixed-size variants (200 tokens / 1,728 chunks, 512 tokens / 676 chunks, 1,000 tokens / 346 chunks) with hit rate bar charts, matching the Session 8 slides 27\u201328 deliverable", font: FONT, size: 22, color: BODY })], { bullet: true }),
      bodyP([new TextRun({ text: "Evaluation rigour \u2014 six diverse GNSS test scenarios with IR metrics (Hit Rate @k, MRR per Session 8 slide 26), granular latency breakdown (embedding/retrieval/generation per slide 22), citation rate tracking, and per-document retrieval frequency analysis", font: FONT, size: 22, color: BODY })], { bullet: true }),
      bodyP([new TextRun({ text: "Interactive Plotly visualizations \u2014 donut chart for latency breakdown, colour-coded relevance bars, grouped strategy comparison charts, hit rate bars, and gradient search latency charts, all matching the dark navy + cyan dashboard theme", font: FONT, size: 22, color: BODY })], { bullet: true }),
      bodyP([new TextRun({ text: "Session concept mapping \u2014 each feature explicitly tied to concepts from Sessions 6, 7, or 8", font: FONT, size: 22, color: BODY })], { bullet: true }),

      bodyP("Revisions were made iteratively until the group agreed that the final work was technically sound, visually polished, and fully responsive to the assignment requirements."),

      // ══════════════════════════════════════════════
      // 7. FINAL SUBMISSION ARTIFACTS
      // ══════════════════════════════════════════════
      heading("7", "Final Submission Artifacts"),

      new Table({
        width: { size: 9840, type: WidthType.DXA },
        columnWidths: [600, 2000, 1200, 6040],
        rows: [
          new TableRow({ children: [
            tc("#", 600, { bold: true, color: "FFFFFF", fill: NAVY, align: AlignmentType.CENTER }),
            tc("Artifact", 2000, { bold: true, color: "FFFFFF", fill: NAVY, align: AlignmentType.CENTER }),
            tc("Format", 1200, { bold: true, color: "FFFFFF", fill: NAVY, align: AlignmentType.CENTER }),
            tc("Description", 6040, { bold: true, color: "FFFFFF", fill: NAVY }),
          ]}),
          ...[
            ["1", "AI Process Log", ".docx", "This document \u2014 records the group\u2019s AI-assisted workflow, tools used, review process, and final deliverables."],
            ["2", "Interactive Dashboard", "Streamlit", "Live web application with three tabs: Q&A (source-cited answers, RAG vs. Plain LLM comparison), Evaluation Dashboard (IR metrics, latency breakdown, Plotly charts), and Strategy Comparison (chunking strategies + chunk-size variants with hit rate bar charts per Session 8 slides 27\u201328)."],
            ["3", "Source Code", "GitHub", "Complete Python codebase: app.py (Streamlit UI), rag_pipeline.py (retrieval & generation), ingest.py (PDF processing & chunking), config.py (configuration), and pre-built ChromaDB vector database."],
          ].map((r, i) => new TableRow({ children: [
            tc(r[0], 600, { fill: i % 2 === 0 ? ALT_ROW : "FFFFFF", align: AlignmentType.CENTER }),
            tc(r[1], 2000, { bold: true, fill: i % 2 === 0 ? ALT_ROW : "FFFFFF" }),
            tc(r[2], 1200, { fill: i % 2 === 0 ? ALT_ROW : "FFFFFF", align: AlignmentType.CENTER }),
            tc(r[3], 6040, { fill: i % 2 === 0 ? ALT_ROW : "FFFFFF", sz: 20 }),
          ]})),
        ],
      }),

      // Dashboard link box
      new Paragraph({ spacing: { before: 160 }, children: [] }),
      new Table({
        width: { size: 9840, type: WidthType.DXA },
        columnWidths: [9840],
        rows: [new TableRow({
          children: [new TableCell({
            borders, width: { size: 9840, type: WidthType.DXA },
            margins: { top: 80, bottom: 80, left: 200, right: 200 },
            children: [
              new Paragraph({ spacing: { after: 20 }, children: [
                new TextRun({ text: "Dashboard Link: ", font: FONT, size: 22, bold: true, color: NAVY }),
                new ExternalHyperlink({ children: [new TextRun({ text: STREAMLIT_URL, font: FONT, size: 22, color: LINK_BLUE, underline: {} })], link: STREAMLIT_URL }),
              ]}),
              new Paragraph({ spacing: { after: 0 }, children: [
                new TextRun({ text: "GitHub Repository: ", font: FONT, size: 22, bold: true, color: NAVY }),
                new ExternalHyperlink({ children: [new TextRun({ text: GITHUB_URL, font: FONT, size: 22, color: LINK_BLUE, underline: {} })], link: GITHUB_URL }),
              ]}),
            ],
          })],
        })],
      }),

      hr(),

      // ══════════════════════════════════════════════
      // 8. CLOSING STATEMENT
      // ══════════════════════════════════════════════
      heading("8", "Closing Statement"),
      bodyP("This assignment was completed through a collaborative group workflow in which AI tools were used as production and analysis support. All final decisions on content selection, technical accuracy, refinement, and submission were made collectively by the group members. The AI tools augmented our workflow but did not replace our critical evaluation and domain judgment."),
      bodyP("The resulting prototype validates the central thesis of Sessions 6 through 8: that foundation models require external grounding mechanisms to produce reliable domain-specific outputs. By implementing a complete RAG pipeline over five GNSS engineering documents, the application demonstrates that hallucination can be systematically mitigated through retrieval-augmented architectural design, producing verifiable, source-cited answers across diverse technical scenarios."),
    ],
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync("Assignment_3_Report.docx", buffer);
  console.log("Report saved: Assignment_3_Report.docx");
});
