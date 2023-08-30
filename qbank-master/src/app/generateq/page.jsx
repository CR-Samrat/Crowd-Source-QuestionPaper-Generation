'use client'
import React, { useState, useEffect } from 'react'
import { parseCookies } from 'nookies'
import Pdfgen from './Pdfgen'
import { PDFDownloadLink, PDFRenderer, PDFViewer } from '@react-pdf/renderer'
import ReactPDF from '@react-pdf/renderer'
import {v4 as uuidv4} from 'uuid'
const subtopicscomputerscience = {
    1: { title: 'computer programming' },
    2: { title: 'Data Structures' },
    3: { title: 'Algorithms' },
    4: { title: 'Operating Systems' },
    5: { title: 'Networking' },
    6: { title: 'Databases' },
    7: { title: 'Computer Architecture' },
    // 8: { title: 'Theory of Computation' }

}
const subtopicsmaths = {
    1: { title: 'Algebra' },
    2: { title: 'Calculus' },
    3: { title: 'Trigonometry' },
    4: { title: 'Geometry' },

}


export default function page() {
    const cookie = parseCookies()
    useEffect(() => {
        if (!cookie.email) {
            window.location.href = '/Signin'
        }
        // const uuid = uuidv4()
        // console.log(uuid)
        // setUuid(uuid)
    }, [])
    const [uuid, setUuid] = useState('');
    const [subject, setSubject] = useState('');
    const [title, setTitle] = useState('')
    const [marks, setMarks] = useState()
    const [selectedSubtopics, setSelectedSubtopics] = useState([]);
    const [showPdf, setShowPdf] = useState(false)
    const [result, setResult] = useState()
    const handleCheckboxChange = (subtopicId) => {
        setSelectedSubtopics((prevSelected) => {
            // Check if the subtopic is already selected
            if (prevSelected.includes(subtopicId)) {
                // Remove the subtopic if it was already selected
                return prevSelected.filter((id) => id !== subtopicId);
            } else {
                // Add the subtopic if it was not already selected
                return [...prevSelected, subtopicId];
            }
        });
    };

    // Handle "Select All" checkbox change
    const handleSelectAllChange = (event) => {
        if (subject == 'computer science') {
            if (event.target.checked) {
                setSelectedSubtopics(Object.keys(subtopicscomputerscience).map(Number));
            } else {
                setSelectedSubtopics([]);
            }
        }
        if (subject == 'maths') {
            if (event.target.checked) {
                setSelectedSubtopics(Object.keys(subtopicsmaths).map(Number));
            } else {
                setSelectedSubtopics([]);
            }
        }
    };
    const handlesubmit = async (e) => {
        
        e.preventDefault()
        if (!title || !marks || !selectedSubtopics || !subject) {
            alert('Please fill all the fields')
            return
        }
        const uuid = uuidv4()
        console.log(uuid)
        setUuid(uuid)
        const mappedTopics = selectedSubtopics.map((key) => subtopicscomputerscience[key]?.title || '');
        console.log(title, marks, selectedSubtopics, subject, mappedTopics)

        const res = await fetch('http://localhost:8080/qms/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ title,subject, topic: mappedTopics, full_marks: marks, email: cookie.email, uuid })
        })
        const json = await res.json()
        if (json)
            console.log(json)
        setResult(json)
        setShowPdf(true)
        setUuid('')
        

    }
    return (
        <div>
            <div className="min-h-screen p-6 bg-gray-100 flex  justify-center">
                <div className="container max-w-screen-lg m-5  mx-auto">
                    <div>
                        {/* <h2 className="font-semibold text-xl text-gray-600">Question details</h2> */}
                        {/* <p className="text-gray-500 mb-6">Form is mobile responsive. Give it a try.</p> */}

                        <div className="bg-white rounded shadow-lg p-4 px-4 md:p-8 mb-6">
                            {!showPdf ? <div className="grid gap-4 gap-y-2 text-sm grid-cols-1 lg:grid-cols-4">
                                <div className="text-gray-600 justify-center">
                                    <p className="font-medium text-lg">Question Paper Details</p>
                                    <p>Please fill out all the fields.</p>
                                </div>

                                <div className="lg:col-span-3">
                                    <div className="grid gap-4 gap-y-2 text-sm grid-cols-1 md:grid-cols-5">
                                        <div className="md:col-span-5">
                                            <label htmlFor="full_name">Subject</label>

                                            <select className='h-10 border mt-1 rounded px-4 w-full bg-gray-50'
                                                onChange={(e) => setSubject(e.target.value)}>
                                                <option value=''>Select Subject</option>
                                                {/* <option value='maths'>Maths</option> */}
                                                <option value='computer science'>Computer Science</option>
                                                {/* <option value='Chemistry'>Chemistry</option> */}
                                            </select>
                                        </div>
                                        <div className="md:col-span-5">
                                            <label htmlFor="full_name">Title of the paper</label>
                                            <input type="text" name="full_name" id="full_name" spellCheck="true" value={title} onChange={(e) => setTitle(e.target.value)}
                                                className="h-10 border mt-1 rounded px-4 w-full bg-gray-50" placeholder='Title of the question paper' />
                                        </div>
                                        <div className="md:col-span-5 flex">
                                            
                                            {subject == 'computer science' ?
                                                <div>
                                                    <label className=' w-24 mt-2'>
                                                        <input
                                                            type="checkbox"
                                                            checked={selectedSubtopics.length === Object.keys(subtopicscomputerscience).length}
                                                            onChange={handleSelectAllChange}
                                                        />{' '}
                                                        Select All
                                                    </label>
                                                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
                                                        {Object.entries(subtopicscomputerscience).map(([id, subtopic]) => (
                                                            <div key={id} className='flex mt-2 ms-2'>
                                                                <label>
                                                                    <input
                                                                        type="checkbox"
                                                                        checked={selectedSubtopics.includes(Number(id))}
                                                                        onChange={() => handleCheckboxChange(Number(id))}
                                                                    />{' '}
                                                                    <span >{subtopic.title}</span>
                                                                </label>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>


                                                : subject == 'maths' ?
                                                    <div>
                                                        <label className=' w-24 mt-2'>
                                                            <input
                                                                type="checkbox"
                                                                checked={selectedSubtopics.length === Object.keys(subtopicsmaths).length}
                                                                onChange={handleSelectAllChange}
                                                            />{' '}
                                                            Select All
                                                        </label>
                                                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
                                                            {Object.entries(subtopicsmaths).map(([id, subtopic]) => (
                                                                <div key={id} className='flex mt-2 ms-2'>
                                                                    <label>
                                                                        <input
                                                                            type="checkbox"
                                                                            checked={selectedSubtopics.includes(Number(id))}
                                                                            onChange={() => handleCheckboxChange(Number(id))}
                                                                        />{' '}
                                                                        <span >{subtopic.title}</span>
                                                                    </label>
                                                                </div>
                                                            ))}
                                                        </div>
                                                    </div> : null
                                            }
                                        </div>
                                        <div className="md:col-span-5">
                                            <label htmlFor="Marks">Marks{' '}</label>
                                            <select className='h-10 border mt-1 rounded px-4 w-full bg-gray-50'
                                                onChange={(e) => setMarks(e.target.value)}>
                                                <option value=''>Total Marks</option>
                                                <option value={30}>30 Marks</option>
                                                <option value={50}>50 Marks</option>
                                                <option value={70}>70 Marks</option>
                                                <option value={100}>100 Marks</option>
                                                {/* <option value='Chemistry'>Chemistry</option> */}
                                            </select>
                                        </div>
                                        <div className="md:col-span-5 text-right">
                                            <div className="inline-flex items-end">
                                                <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                                                    onClick={handlesubmit}
                                                >
                                                    Submit</button>
                                            </div>
                                        </div>

                                    </div>
                                </div>
                            </div> : <div className='justify-center'>
                                <p className='text-xl font-bold'>Question Paper Generated Successfully</p>
                                {
                                    showPdf ? <PDFDownloadLink document={<Pdfgen title={title} subject={subject} marks={marks} data ={result} />} fileName={title}>
                                        {({ blob, url, loading, error }) => (loading ? <h4>Loading Document</h4>:<h4 className='text-blue-500 text-lg'>Download Now</h4>)}
                                    </PDFDownloadLink> : null
                                }
                            </div>
                            }
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}
